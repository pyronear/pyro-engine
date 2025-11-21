![PyroNear Logo](docs/source/_static/img/pyronear-logo-dark.png)

<p align="center">
  <a href="https://github.com/pyronear/pyro-engine/actions?query=workflow%3Atests">
    <img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/pyronear/pyro-engine/tests.yml?branch=develop&label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://github.com/pyronear/pyro-engine/actions?query=workflow%3Adocs">
    <img src="https://img.shields.io/github/actions/workflow/status/pyronear/pyro-engine/docs.yml?branch=main&label=docs&logo=read-the-docs&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/pyronear/pyro-engine">
    <img src="https://img.shields.io/codecov/c/github/pyronear/pyro-engine.svg?logo=codecov&style=flat-square" alt="Test coverage percentage">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://www.codacy.com/gh/pyronear/pyro-engine/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyronear/pyro-engine&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/108f5fe8a7ac4f40a7bbd1985e26d5f9"/></a>
</p>
<p align="center">
  <a href="https://pypi.org/project/pyroengine/">
    <img src="https://img.shields.io/pypi/v/pyroengine.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPi Version">
  </a>
  <a href="https://hub.docker.com/r/pyronear/pyro-engine">
    <img alt="DockerHub version" src="https://img.shields.io/docker/v/pyronear/pyro-engine/latest?label=Docker&logo=Docker&logoColor=white">
  </a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square&logo=python&logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/pyroengine.svg?style=flat-square" alt="license">
</p>

# PyroEngine repository

This repository contains two main components used in Pyronear wildfire detection deployments.

| Directory          | Description                                                               |
| ------------------ | ------------------------------------------------------------------------- |
| `pyroengine/`      | Detection engine that runs the wildfire model on edge devices             |
| `pyro_camera_api/` | Camera control API that exposes a unified interface for multiple backends |
| `scripts/`         | Shell scripts for deployment and debugging                                |
| `src/`             | Helper scripts for camera control, focus, calibration and experiments     |
| `tests/`           | Test suite for the detection engine                                       |

Both components can run independently or together in the same deployment.

---

## PyroEngine: wildfire detection on edge devices

PyroEngine provides a high level interface to use deep learning models in production while being connected to the alert API.

### Quick example

```python
from pyroengine.core import Engine
from PIL import Image

engine = Engine()

im = Image.open("path/to/your/image.jpg").convert("RGB")

prediction = engine.predict(im)
```

---

## PyroCamera API: unified camera control

The `pyro_camera_api` package provides a REST API and a Python client to control cameras and retrieve images in a unified way.

The API supports multiple camera backends through a common abstraction:

* `reolink` backend, for Reolink PTZ or static cameras
* `rtsp` backend, for RTSP streams
* `url` backend, for HTTP snapshot URLs
* `fake` backend, for development and tests

Each backend has its own class and inherits from the same base interface. The system selects the correct implementation at runtime based on the `backend` field in `credentials.json`. The API routes are the same for all camera types. PTZ cameras use pose and movement, other cameras ignore pose parameters without failing.

### Simple API usage example

```python
from pyro_camera_api_client import Client

client = Client("http://localhost:8000")

# Move a PTZ camera to a pose
client.move("ptz_camera_1", pose=2)

# Get a snapshot as bytes
image_bytes = client.snapshot("url_camera_1")
```

---

## Setup

Python 3.11 or higher and `pip` or `conda` are required.

### Developer installation

```bash
git clone https://github.com/pyronear/pyro-engine.git
pip install -e pyro-engine/.
```

### Environment variables

Deployments usually rely on a `.env` file with information such as:

```text
API_URL=https://api.pyronear.org
CAM_USER=my_dummy_login
CAM_PWD=my_dummy_pwd
```

### Data directory

A `./data` directory is expected with at least:

* `credentials.json`
* optionally `model.onnx` to override weights from Hugging Face
* optionally `config.json` to override model configuration

---

## Camera configuration and backends

The camera configuration is stored in `credentials.json`. Each key represents a camera identifier and the entry defines how to access and control it.

The important field for the camera API is:

* `backend` which selects the camera implementation

Other fields such as `type`, `azimuth`, `poses`, or `bbox_mask_url` are used by the engine and the API.

Below is one generic example for each backend: `url`, `rtsp`, `reolink` static, `reolink` PTZ and `fake`.

```json
{
  "url_camera_1": {
    "name": "url_camera_1",
    "backend": "url",
    "url": "http://user:password@camera-host:1234/cgi-bin/snapshot.cgi",
    "azimuth": 0,
    "id": "10",
    "bbox_mask_url": "",
    "poses": [],
    "token": "JWT_TOKEN_HERE",
    "type": "static"
  },

  "rtsp_camera_1": {
    "name": "rtsp_camera_1",
    "backend": "rtsp",
    "rtsp_url": "rtsp://user:password@camera-host:554/live/STREAM_ID",
    "azimuth": 0,
    "id": "11",
    "bbox_mask_url": "https://example.com/occlusion-masks/rtsp_camera_1",
    "poses": [],
    "token": "JWT_TOKEN_HERE",
    "type": "static"
  },

  "reolink_static_1": {
    "name": "reolink_static_1",
    "backend": "reolink",
    "type": "static",
    "azimuth": 45,
    "id": "12",
    "poses": [],
    "bbox_mask_url": "https://example.com/occlusion-masks/reolink_static_1",
    "token": "JWT_TOKEN_HERE"
  },

  "reolink_ptz_1": {
    "name": "reolink_ptz_1",
    "backend": "reolink",
    "type": "ptz",
    "id": "13",
    "poses": [0, 1, 2, 3],
    "azimuths": [0, 90, 180, 270],
    "bbox_mask_url": "https://example.com/occlusion-masks/reolink_ptz_1",
    "token": "JWT_TOKEN_HERE"
  },

  "fake_camera_1": {
    "name": "fake_camera_1",
    "backend": "fake",
    "type": "static",
    "azimuth": 0,
    "id": "14",
    "poses": [],
    "bbox_mask_url": "",
    "token": "JWT_TOKEN_HERE"
  }
}
```

These examples are only for illustration. Real deployments use real URLs and real tokens generated by the alert API.

---

## Documentation

The full package documentation is available here:

[https://pyronear.org/pyro-engine/](https://pyronear.org/pyro-engine/)

It covers engine usage, configuration and deployment details.

---

## Contributing

Please refer to [`CONTRIBUTING`](CONTRIBUTING.md) if you wish to contribute to this project.

---

## License

Distributed under the Apache 2 License. See [`LICENSE`](LICENSE) for more information.
