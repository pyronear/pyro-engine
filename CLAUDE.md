# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyroEngine is a wildfire detection system for edge devices (Raspberry Pi, etc.). It has three packages:

- **`pyroengine/`** — Core detection engine: runs YOLO model inference on camera images, manages alert states, communicates with the PyroNear API.
- **`pyro_camera_api/`** — FastAPI service: unified REST interface for controlling heterogeneous cameras (Reolink, Linovision/Hikvision, RTSP, HTTP URL, generic REST/JSON snapshot API).
- **`pyro_temporal_api/`** — Optional FastAPI service: validates ongoing alerts with the temporal smoke model (`pyronear/temporal-model`, ONNX runtime, no torch) before the engine sends them to the PyroNear API.

These services run as separate Docker containers and communicate over localhost (host network mode). The engine calls the camera API to capture frames and manage PTZ patrols.

## Common Commands

```bash
# Style / lint
make style        # auto-fix: ruff format + ruff check --fix
make quality      # check-only: ruff format, ruff check, mypy

# Tests
make test         # pytest with coverage

# Docker (production)
make run          # build both images and start docker-compose stack
make stop         # stop stack
make log          # tail engine logs
make log-api      # tail camera API logs
```

Run a single test file:
```bash
pytest tests/test_engine.py -v
```

## Architecture

### Detection Flow

1. `SystemController` (`pyroengine/core.py`) orchestrates the main loop: captures frames from each camera via the Camera API, feeds them to the `Engine`.
2. `Engine` (`pyroengine/engine.py`) runs inference via `Classifier` and maintains a per-camera sliding window of predictions. When confidence exceeds a threshold it fires an alert to the PyroNear API.
3. `Classifier` (`pyroengine/vision.py`) wraps a YOLO11 model, using NCNN (ARM-optimized) or ONNX. Model weights are auto-downloaded from Hugging Face Hub on first run.

### Camera API

- Entry point: `pyro_camera_api/pyro_camera_api/main.py` (FastAPI + lifespan)
- Camera adapters in `camera/adapters/`: `reolink.py`, `linovision.py`, `hikvision.py`, `rtsp.py`, `url.py`, `rest.py`, `mock.py` — all inherit from abstract bases in `camera/base.py`
  - `hikvision.py` (`HikvisionCamera`, adapter `"hikvision"`) is separate from Linovision: ISAPI `absoluteEx` in decimal degrees and zoom ratio (not tenths); zoom range and reserved preset ids are read from `/ISAPI/PTZCtrl/channels/{ch}/capabilities`.
  - `rest.py` (`RestSnapshotCamera`, adapter `"rest"`/`"api"`) is a config-driven HTTP snapshot adapter for endpoints that need custom auth headers or return the image wrapped in JSON (base64 or nested URL), e.g. vigilant.cat. Header/URL values may reference env vars via `${VAR}`.
- `camera/registry.py` tracks live camera instances and background threads
- Background patrol loops run in `camera/patrol.py`
- Routes under `api/`: cameras, control (PTZ), focus, patrol, stream, health

### Data / Config (mounted at `./data/`)

- `credentials.json` — camera list and credentials (required). Schema (keyed by camera IP):
  ```json
  {
    "192.168.1.10": {
      "token": "<JWT>",
      "type": "ptz",          // or "static"
      "name": "site-cam-01",
      "brand": "reolink",     // or "linovision", "hikvision"
      "id": 7,                // camera ID on pyronear API
      "poses": [0, 1, 2, 3], // PTZ preset IDs; empty list for static
      "azimuths": [0, 90, 180, 270]
    }
  }
  ```
- `model.onnx` — optional custom model weights
- `config.json` — optional custom model config

### `cam_id` naming convention

`Engine` and `SystemController` identify each camera-pose pair by a string `cam_id`:
- PTZ camera at pose `p`: `"{ip}_{p}"` (e.g. `"192.168.1.10_2"`)
- Static camera: `"{ip}"` (e.g. `"192.168.1.11"`)

`Engine.__init__` receives `cam_creds` as `Dict[cam_id, Tuple[token, pose_id, bbox_mask_url]]`, which is a flattened/transformed form of `credentials.json` (one entry per pose, not per IP). The transformation from raw JSON to this format happens in the entrypoint script, not inside the library.

### Occlusion masks

For each `cam_id`, the engine periodically fetches a JSON file at `{bbox_mask_url}_{pose_id}.json` from a remote URL to get a dict of bounding boxes marking permanently occluded regions. Predictions with IoU > 0.1 against any occlusion box are dropped before confidence scoring.

### Temporal validation (optional)

The service is behind the compose profile `temporal` (`COMPOSE_PROFILES=temporal` in `.env`). When `TEMPORAL_API_URL` is set, `Engine` widens its per-`cam_id` prediction window to `temporal_window` (11) frames (JPEG + YOLO boxes); detection still pools the last `nb_consecutive_frames`. While an alert is ongoing, `Engine._temporal_gate` submits that window to `pyro_temporal_api` every round (`POST /jobs`, returns immediately, at most `MAX_FRAMES` = 11 frames) and reads the verdict on the next round (`GET /jobs/{id}`). The verdict lists the scored tubes (union box + probability); boxes covered by a tube scoring above `temporal_min_score` (0.2) are uploaded with that probability as `temporal_model_scores` (pyro-api PR 723 applies its own threshold instead of scoring the sequence), boxes outside every scored tube are dropped. Only the frames a verdict scored are uploaded on it, later frames wait for the next verdict; a verdict without such tube stages nothing (frames wait for the next one); a pending job holds the alert one more round; frames of an ended event rejected by the model are dropped for good. Service errors, a lost job and a job pending longer than `temporal_job_timeout` (120 s) fail open: the rest of the event is sent without scores and the API scores it itself. `/health` returns 503 when the scoring worker died or is stuck. The service loads `data/model_onnx.zip` (`TEMPORAL_MODEL_PATH`), exported from a temporal-model release with `temporal-export-onnx`. A format 2 package (split `backbone.onnx` + `head.onnx`) lets the service embed only real frames and cache per-slot ViT embeddings across jobs (`EMBEDDING_CACHE_SIZE` in `main.py`, keyed by frame id + crop box); the verdict reports `embeddings_cached` / `embeddings_new` per job.

### Stream-awareness

`SystemController.inference_loop` calls `_any_stream_active()` before and during every camera loop. If an active RTSP/SRT pipeline is detected (via `/stream/status`), the entire inference pass is skipped to avoid interfering with live streaming.

### Night / day handling

Day is determined by IR-channel analysis (`is_day_time(strategy="ir")`): if `max(R - G) == 0` the image is considered infrared/night. At night the main loop sleeps for 1 hour, stops all patrols, then re-checks before resuming inference.

### Environment Variables (`.env`)

Key vars used at runtime: `LAT`, `LON`, `API_URL`, `API_TOKEN`, `CAM_USER`, `CAM_PWD`, `MEDIAMTX_SERVER_IP`, `ROUTER_IP`, `ROUTER_USER`, `ROUTER_PASSWORD`, `ENABLE_ROUTER_REBOOT`, `COMPOSE_PROFILES=temporal` + `TEMPORAL_API_URL` (e.g. `http://localhost:8082`; both unset = no temporal validation), `TEMPORAL_MODEL_PATH`.

### Legacy direct-camera module

`pyroengine/sensors.py` contains `ReolinkCamera`, a standalone class that talks directly to Reolink cameras via their HTTP CGI API (no Camera API service). It is used by utility/debug scripts, not by the main detection pipeline.

## Code Style

- Python 3.11+, line length 120
- Ruff for formatting and linting; mypy with strict checks
- Test coverage enforced via `pytest-cov`
- License headers are validated in CI (`style.yml`)
