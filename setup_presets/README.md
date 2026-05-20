# setup_presets

One-off tool that registers Reolink PTZ presets `0`, `1`, `2`, `3` spaced
~45° apart, starting from a manually configured reference preset `10`.

It talks directly to each camera's `cgi-bin/api.cgi` — no Pyro camera API
service required.

## Setup

Copy `.env.example` to `.env` and fill in the camera credentials:

```
CAM_USER=admin        # camera login (same user for every IP passed on the CLI)
CAM_PWD=your_password # camera password
```

Both variables are required; the script exits early if either is missing.

## Run

```bash
uv run python setup_presets.py 192.168.1.11 192.168.1.12
```

Pass `--protocol http` if your cameras don't accept HTTPS.

## What it does for each camera

1. Verifies reference preset `10` is configured (skips with a warning otherwise).
2. Moves to preset `10`.
3. Pans 90° left to the leftmost target position.
4. Saves preset `0`, then steps 45° right between each of presets `1`, `2`, `3`.
