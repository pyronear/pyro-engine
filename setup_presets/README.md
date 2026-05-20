# setup_presets

One-off tool that registers Reolink PTZ presets `0`, `1`, `2`, `3` spaced
~45° apart, starting from wherever the camera is currently pointing. Aim
each camera at the desired centre position by hand (or via a preset)
before running.

Cameras passed on the CLI are processed in parallel. It talks directly
to each camera's `cgi-bin/api.cgi` — no Pyro camera API service required.

## Setup

Copy `.env.example` to `.env` and fill in the camera credentials:

```
CAM_USER=admin        # camera login (same user for every IP passed on the CLI)
CAM_PWD=your_password # camera password
```

Both variables are required; the script exits early if either is missing.

## Run

From this folder (`cd setup_presets`):

```bash
# Register presets 0-3 (overwrites any existing ones)
uv run python setup_presets.py 192.168.1.11 192.168.1.12

# Patrol the registered presets (3 cycles by default, 3s dwell per pose)
uv run python patrol_presets.py 192.168.1.11 192.168.1.12

# Override the cycle count with -n
uv run python patrol_presets.py 192.168.1.11 192.168.1.12 -n 10
```

Pass `--protocol http` if your cameras don't accept HTTPS.

## What it does

`setup_presets.py` — for each camera (run in parallel):

1. Pans 90° left from the current position to the leftmost target.
2. Saves preset `0`, then steps 45° right between each of presets `1`, `2`, `3`.

`patrol_presets.py` — for each camera (run in parallel): cycles through
presets `0 → 1 → 2 → 3` with a 3-second dwell at each pose, repeated `-n`
times (default 3).
