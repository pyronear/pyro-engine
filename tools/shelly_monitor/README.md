# shelly_monitor

Collect and analyze power consumption data from a **Shelly Gen2** device via its local HTTP API.

Compatible with any Gen2 Shelly device equipped with a Switch PM component.
The number of available channels depends on the device model — refer to your
device's documentation or the Shelly app to find the right value.

---

## Structure

```
tools/shelly_monitor/
|-- collect.py    # Periodic data collection (CSV)
|-- analyze.py    # Analysis and report generation
|-- README.md
```

---

## Requirements

No mandatory external dependencies - both scripts use the Python standard library only.

For the optional chart (`--plot`):
```bash
apt install python3-matplotlib
```

---

## Configuration

### Channel names

Edit the `CHANNEL_NAMES` dict at the top of `collect.py` to match your physical setup:

```python
CHANNEL_NAMES = {
    0: "rpi",           # channel 0 label in the CSV
    1: "router_switch", # channel 1 label in the CSV (second channel, if available)
}
```

### Finding your Shelly's IP address

- Shelly app -> Device -> Local IP
- Router DHCP table: look for a device named example : `ShellyPro2PM-XXXXXX`
- Quick test:
  ```bash
  curl http://<SHELLY_IP>/rpc/Switch.GetStatus?id=0
  ```

---

## Data collection (on the Raspberry Pi)

### One-shot run
```bash
python3 collect.py --host 192.168.1.42
python3 collect.py --host 192.168.1.42 --channels 1 --output /data/shelly_data.csv
```

### Cron (every 10 minutes)
```bash
crontab -e
```
Add:
```
*/10 * * * * /usr/bin/python3 /path/to/tools/shelly_monitor/collect.py --host 192.168.1.90 --channels 1 --output /data/shelly_data.csv >> /data/collect.log 2>&1
```

Comment the line with '#' if you want to stop the cron.

### Help
```
python3 collect.py --help
```

---

## Analysis (on your PC or the RPi)

### Retrieve the CSV from the Raspberry Pi
```bash
scp pi@<RPI_IP>:/data/shelly_data.csv .
```

### Run the analysis
```bash
# Text report only
python3 analyze.py --input shelly_data.csv

# With a power-over-time chart (saved as shelly_power.png)
python3 analyze.py --input shelly_data.csv --plot

# Custom night window (default: 20h -> 6h)
python3 analyze.py --input shelly_data.csv --night-start 19 --night-end 7

# Custom night window (default: 20h -> 6h) with a power-over-time chart
python3 analyze.py --input shelly_data.csv --night-start 19 --night-end 7 --plot
```

### Help
```
python3 analyze.py --help
```

---

## CSV format

| Column | Description |
|---|---|
| `timestamp` | Measurement timestamp (`YYYY-MM-DD HH:MM:SS`) |
| `channel_id` | Channel index (0 or 1) |
| `channel_name` | Human-readable channel label |
| `output` | Relay state (`True` = ON, `False` = OFF) |
| `power_w` | Instantaneous active power (W) |
| `voltage_v` | RMS voltage (V) |
| `current_a` | RMS current (A) |
| `pf` | Power factor (0.0-1.0) |
| `freq_hz` | AC frequency (Hz) |
| `energy_wh` | Cumulative energy counter (Wh, resets on `Switch.ResetCounters`) |
| `energy_by_minute_mWh` | Energy per minute over the last 3 minutes (JSON list, milliwatt-hours) |
| `temperature_c` | Internal device temperature (°C) |

---

## Report metrics

- Total energy and normalised energy per 24 h (kWh)
- Average, peak, and minimum instantaneous power
- Average power during day vs night (configurable fixed hours)
- Combined totals across all channels
- Optional power-over-time chart with night periods shaded

---

## Notes

- Energy (kWh) is estimated by integrating instantaneous power (`apower`) over time. The average
  interval between consecutive samples is computed from the timestamps and used as Δt.
- The `energy_wh` column stores the device's own cumulative counter (`aenergy.total`) and can be
  used for cross-validation against the integrated estimate.
- The day/night boundary is defined by fixed hours. Once you have several days of data, inspect
  the per-channel consumption curves to identify the actual camera shutdown time and refine
  `--night-start` / `--night-end` accordingly.
