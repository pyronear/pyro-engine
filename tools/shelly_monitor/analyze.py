#!/usr/bin/env python3
"""
analyze.py - Analyze power consumption data collected by collect.py.

Reads a CSV file produced by collect.py, computes per-channel and combined
statistics, and optionally generates a power-over-time plot.

Day/night split is based on fixed hours (configurable via --night-start /
--night-end). The default window (20h-6h) matches a typical camera deployment
where images are not captured after dark. Adjust it to match your observations
once you have a few days of data.

Usage:
  python analyze.py --input shelly_data.csv
  python analyze.py --input shelly_data.csv --plot
  python analyze.py --input shelly_data.csv --night-start 19 --night-end 7

Typical workflow when running collect.py on a Raspberry Pi:
  scp pi@<RPI_IP>:/path/to/shelly_data.csv .
  python analyze.py --input shelly_data.csv --plot
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta

# - Day/night defaults ----------------------------
# Night = period when cameras are inactive (insufficient light).
# These are initial estimates; refine after inspecting a few days of data.
DEFAULT_NIGHT_START = 20   # 8 PM
DEFAULT_NIGHT_END   = 6    # 6 AM
# ---------------------------------------


def load_csv(filepath: str) -> list[dict]:
    """Load and parse a CSV file produced by collect.py.

    Converts numeric fields from strings to float (None on missing/invalid
    values) and parses the timestamp into a datetime object stored under the
    '_dt' key (used internally; not present in the original CSV).
    Rows with an unparseable timestamp are silently dropped.

    Args:
        filepath: Path to the CSV file to read.

    Returns:
        List of row dicts with typed fields and an added '_dt' key.
    """
    numeric_fields = ("power_w", "voltage_v", "current_a", "pf", "freq_hz",
                      "energy_wh", "temperature_c")
    rows = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric fields; treat empty strings and "None" as missing
            for field in numeric_fields:
                try:
                    row[field] = float(row[field]) if row[field] not in ("", "None") else None
                except (ValueError, TypeError):
                    row[field] = None

            row["channel_id"] = int(row["channel_id"]) if row["channel_id"] else 0

            # Parse timestamp into a datetime for time-based calculations
            try:
                row["_dt"] = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                row["_dt"] = None

            rows.append(row)

    # Drop rows where the timestamp could not be parsed
    return [r for r in rows if r["_dt"] is not None]


def is_night(dt: datetime, night_start: int, night_end: int) -> bool:
    """Return True if the given datetime falls within the night window.

    Handles windows that span midnight (e.g. 20h -> 6h) by checking whether
    the hour is >= night_start OR < night_end in that case.

    Args:
        dt:          Datetime to classify.
        night_start: Hour (0-23) at which night begins (inclusive).
        night_end:   Hour (0-23) at which night ends (exclusive).

    Returns:
        True if the hour falls within the night window, False otherwise.
    """
    h = dt.hour
    if night_start > night_end:
        # Window crosses midnight: e.g. 20h-6h
        return h >= night_start or h < night_end
    else:
        # Window within a single day: e.g. 0h-6h
        return night_start <= h < night_end


def group_by_channel(rows: list[dict]) -> dict[str, list[dict]]:
    """Group rows by channel name.

    Args:
        rows: Flat list of row dicts from load_csv().

    Returns:
        Dict mapping channel_name -> list of rows for that channel.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["channel_name"]].append(r)
    return dict(groups)


def compute_energy_kwh(rows: list[dict]) -> float:
    """Estimate total energy consumption in kWh from instantaneous power samples.

    Uses a simple rectangular integration: energy = sum(P_i * Δt), where Δt
    is the average interval between consecutive measurements (in hours).
    This is accurate when the sampling interval is consistent (e.g. every 10 min).

    Args:
        rows: List of row dicts for a single channel, in any order.

    Returns:
        Estimated energy in kWh, or 0.0 if fewer than 2 valid samples exist.
    """
    if not rows:
        return 0.0

    powers = [r["power_w"] for r in rows if r["power_w"] is not None]
    if len(powers) < 2:
        return 0.0

    # Compute the average gap between consecutive timestamps
    dts = sorted(r["_dt"] for r in rows if r["_dt"] is not None)
    intervals_h = [(dts[i + 1] - dts[i]).total_seconds() / 3600 for i in range(len(dts) - 1)]
    avg_interval_h = sum(intervals_h) / len(intervals_h) if intervals_h else (10 / 60)

    # Multiply average power by total time covered
    return sum(powers) * avg_interval_h / 1000


def analyze_channel(name: str, rows: list[dict], night_start: int, night_end: int) -> dict:
    """Compute summary statistics for a single channel.

    Splits samples into day/night buckets using is_night(), then computes
    power averages and total energy for each period and overall.

    Args:
        name:        Channel name (used as label in the report).
        rows:        All rows for this channel, from group_by_channel().
        night_start: Hour at which night begins (passed to is_night()).
        night_end:   Hour at which night ends (passed to is_night()).

    Returns:
        Dict of statistics ready for print_report() and plot_data().
    """
    powers = [r["power_w"] for r in rows if r["power_w"] is not None]

    # Split into day vs night subsets
    day_rows   = [r for r in rows if not is_night(r["_dt"], night_start, night_end)]
    night_rows = [r for r in rows if     is_night(r["_dt"], night_start, night_end)]
    day_powers   = [r["power_w"] for r in day_rows   if r["power_w"] is not None]
    night_powers = [r["power_w"] for r in night_rows if r["power_w"] is not None]

    # When rows exist for a period but all power values are None, the device was
    # off (output=False) for that entire window -> treat as 0 W, not missing data.
    # None is reserved for "no rows at all in this period" (truly unknown).
    def avg_power(powers: list, rows: list):
        if powers:
            return round(sum(powers) / len(powers), 2)
        if rows:   # rows exist but device was off -> 0 W
            return 0.0
        return None  # no data at all for this period

    # Total time span covered by the dataset
    dts = sorted(r["_dt"] for r in rows)
    span_hours = (dts[-1] - dts[0]).total_seconds() / 3600 if len(dts) > 1 else 0

    energy_kwh    = compute_energy_kwh(rows)
    # Normalise to a per-24h figure if more than one day of data is available
    energy_per_24h = (energy_kwh / span_hours * 24) if span_hours > 0 else 0

    return {
        "channel":           name,
        "n_samples":         len(rows),
        "span_hours":        round(span_hours, 1),
        "span_days":         round(span_hours / 24, 2),
        "energy_total_kwh":  round(energy_kwh, 4),
        "energy_per_24h_kwh": round(energy_per_24h, 4),
        "avg_power_w":       round(sum(powers) / len(powers), 2)       if powers       else None,
        "max_power_w":       round(max(powers), 2)                     if powers       else None,
        "min_power_w":       round(min(powers), 2)                     if powers       else None,
        "avg_day_power_w":   avg_power(day_powers,   day_rows),
        "avg_night_power_w": avg_power(night_powers, night_rows),
        "n_day_samples":     len(day_rows),
        "n_night_samples":   len(night_rows),
    }


def print_report(stats: list[dict], night_start: int, night_end: int) -> None:
    """Print a formatted text report to stdout.

    Displays per-channel statistics followed by a combined total when more
    than one channel is present.

    Args:
        stats:       List of stat dicts from analyze_channel().
        night_start: Night start hour, shown in the report header.
        night_end:   Night end hour, shown in the report header.
    """
    sep = "-" * 60
    print(f"\n{'=' * 60}")
    print("  POWER CONSUMPTION REPORT - Shelly Monitor")
    print(f"{'=' * 60}")
    print(f"  Night window: {night_start:02d}h -> {night_end:02d}h\n")

    for s in stats:
        print(sep)
        print(f"  Channel: {s['channel']}")
        print(sep)
        print(f"  Period covered       : {s['span_days']} days ({s['span_hours']} h)")
        print(f"  Samples              : {s['n_samples']} data points")
        print()
        print(f"  ⚡ Average power     : {s['avg_power_w']} W")
        print(f"  ⚡ Peak power        : {s['max_power_w']} W")
        print(f"  ⚡ Min power         : {s['min_power_w']} W")
        print()
        print(f"  🌞 Avg DAY power     : {s['avg_day_power_w']} W  ({s['n_day_samples']} samples)")
        print(f"  🌙 Avg NIGHT power   : {s['avg_night_power_w']} W  ({s['n_night_samples']} samples)")
        print()
        print(f"  🔋 Total energy      : {s['energy_total_kwh']} kWh")
        print(f"  🔋 Energy per 24h    : {s['energy_per_24h_kwh']} kWh/day")

    # Combined totals only make sense when multiple channels are present
    if len(stats) > 1:
        total_energy = sum(s["energy_total_kwh"] for s in stats)
        total_24h    = sum(s["energy_per_24h_kwh"] for s in stats)
        avg_powers   = [s["avg_power_w"] for s in stats if s["avg_power_w"] is not None]
        print(f"\n{sep}")
        print("  COMBINED TOTAL (all channels)")
        print(sep)
        print(f"  ⚡ Total avg power   : {round(sum(avg_powers), 2)} W")
        print(f"  🔋 Total energy      : {round(total_energy, 4)} kWh")
        print(f"  🔋 Energy per 24h    : {round(total_24h, 4)} kWh/day")

    print(f"\n{'=' * 60}\n")


def plot_data(rows_by_channel: dict, night_start: int, night_end: int) -> None:
    """Generate and save a power-over-time PNG chart.

    Creates one subplot per channel, with the night window shaded in the
    background. Requires matplotlib; prints a warning and returns gracefully
    if it is not installed.

    Args:
        rows_by_channel: Dict from group_by_channel() - channel_name -> rows.
        night_start:     Night start hour for background shading.
        night_end:       Night end hour for background shading.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("[WARN] matplotlib is not installed. Run: apt install python3-matplotlib")
        return

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]
    n = len(rows_by_channel)
    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]  # ensure axes is always iterable

    for ax, (name, rows), color in zip(axes, rows_by_channel.items(), colors):
        rows_sorted = sorted(rows, key=lambda r: r["_dt"])
        dts    = [r["_dt"] for r in rows_sorted]
        powers = [r["power_w"] if r["power_w"] is not None else 0 for r in rows_sorted]

        ax.plot(dts, powers, color=color, linewidth=1, label=name)
        ax.fill_between(dts, powers, alpha=0.15, color=color)
        ax.set_ylabel("Power (W)")
        ax.set_title(f"Channel: {name}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Shade night periods across all days covered by the dataset
        if dts:
            current = dts[0].date()
            end_day = dts[-1].date() + timedelta(days=1)
            while current < end_day:
                if night_start > night_end:
                    # Window spans midnight: shade from night_start to next day night_end
                    ns = datetime(current.year, current.month, current.day, night_start)
                    ne = datetime(current.year, current.month, current.day, night_end) + timedelta(days=1)
                else:
                    ns = datetime(current.year, current.month, current.day, night_start)
                    ne = datetime(current.year, current.month, current.day, night_end)
                ax.axvspan(ns, ne, alpha=0.08, color="navy")
                current += timedelta(days=1)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d %Hh"))
    fig.autofmt_xdate()
    plt.suptitle("Shelly Power Monitor - Instantaneous Power", fontsize=13, fontweight="bold")
    plt.tight_layout()

    outfile = "shelly_power.png"
    plt.savefig(outfile, dpi=150)
    print(f"Chart saved: {outfile}")
    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Parsed argument namespace. Exits with help on invalid input.
    """
    parser = argparse.ArgumentParser(
        prog="analyze.py",
        description=(
            "Analyze power consumption data collected by collect.py. "
            "Produces a text report with per-channel and combined statistics. "
            "Optionally generates a power-over-time PNG chart (requires matplotlib)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python analyze.py --input shelly_data.csv\n"
            "  python analyze.py --input shelly_data.csv --plot\n"
            "  python analyze.py --input shelly_data.csv --night-start 19 --night-end 7\n\n"
            "retrieve data from the Raspberry Pi first:\n"
            "  scp pi@<RPI_IP>:/path/to/shelly_data.csv ."
        ),
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="FILE",
        help="Path to the CSV file produced by collect.py (required).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a power-over-time PNG chart (requires matplotlib).",
    )
    parser.add_argument(
        "--night-start",
        type=int,
        default=DEFAULT_NIGHT_START,
        metavar="HOUR",
        help=(
            f"Hour (0-23) at which night begins, inclusive "
            f"(default: {DEFAULT_NIGHT_START}). "
            "Night is when cameras stop capturing due to darkness."
        ),
    )
    parser.add_argument(
        "--night-end",
        type=int,
        default=DEFAULT_NIGHT_END,
        metavar="HOUR",
        help=(
            f"Hour (0-23) at which night ends, exclusive "
            f"(default: {DEFAULT_NIGHT_END}). "
            "Supports windows that cross midnight (e.g. 20 -> 6)."
        ),
    )
    args = parser.parse_args()

    # Validate hour ranges
    for attr, label in (("night_start", "--night-start"), ("night_end", "--night-end")):
        val = getattr(args, attr)
        if not (0 <= val <= 23):
            parser.error(f"{label} must be between 0 and 23, got {val}.")

    return args


def main() -> None:
    args = parse_args()

    print(f"Loading {args.input} ...")
    rows = load_csv(args.input)
    print(f"{len(rows)} valid rows loaded.")

    if not rows:
        print("[ERROR] No valid data found in the file.")
        return

    rows_by_channel = group_by_channel(rows)
    stats = [
        analyze_channel(name, ch_rows, args.night_start, args.night_end)
        for name, ch_rows in rows_by_channel.items()
    ]

    print_report(stats, args.night_start, args.night_end)

    if args.plot:
        plot_data(rows_by_channel, args.night_start, args.night_end)


if __name__ == "__main__":
    main()
