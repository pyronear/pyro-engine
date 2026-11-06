# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Exercise locked dependencies, native libraries and an unprivileged read-only runtime."""

import errno
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from http.client import HTTPConnection
from io import BytesIO
from pathlib import Path

import cv2
import ncnn
import numpy as np
import onnxruntime
from packaging.markers import Marker, default_environment
from packaging.version import Version
from PIL import Image


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def check_locked_packages(mode):
    lock = tomllib.loads(Path("/runtime.lock").read_text())
    packages = {normalize(package["name"]): package for package in lock["package"]}
    root = "pyroengine" if mode == "engine" else "pyro-camera-api"
    pending = [(root, ())]
    visited = set()
    expected = {}
    while pending:
        name, extras = pending.pop()
        if (name, extras) in visited or name == "opencv-python":
            continue
        visited.add((name, extras))
        package = packages[name]
        if "virtual" not in package["source"]:
            expected[name] = package.get("version")
        dependencies = [(dependency, "") for dependency in package.get("dependencies", [])]
        for extra in extras:
            dependencies.extend(
                (dependency, extra) for dependency in package.get("optional-dependencies", {}).get(extra, [])
            )
        for dependency, extra in dependencies:
            environment = {**default_environment(), "extra": extra}
            if "marker" not in dependency or Marker(dependency["marker"]).evaluate(environment):
                pending.append((normalize(dependency["name"]), tuple(dependency.get("extra", []))))
    distributions = list(metadata.distributions())
    installed = {normalize(distribution.metadata["Name"]): distribution.version for distribution in distributions}
    assert len(distributions) == len(installed), "Duplicate package metadata in runtime"
    assert installed.keys() == expected.keys(), (installed.keys() - expected.keys(), expected.keys() - installed.keys())
    for name, version in installed.items():
        if expected[name] is not None:
            assert Version(version) == Version(expected[name]), (name, version, expected[name])
    return installed


def check_camera():
    from pyro_camera_api.main import app

    assert "/health" in app.openapi()["paths"]
    assert shutil.which("ffmpeg")
    assert shutil.which("ffprobe")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-filter_threads",
            "1",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x24:rate=1",
            "-frames:v",
            "1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            "-y",
            "/tmp/container-check.mp4",
        ],
        check=True,
    )
    protocols = subprocess.check_output(["ffmpeg", "-hide_banner", "-protocols"], text=True)
    assert {"srt", "rtp"}.issubset(protocols.split())
    Image.new("RGB", (32, 24)).save("/usr/src/app/focus_debug/container-check.jpg")

    # Avoid downloading model weights or contacting cameras in the server smoke check.
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "pyro_camera_api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8081",
            "--lifespan",
            "off",
        ],
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            assert server.poll() is None, "Camera HTTP server stopped"
            connection = HTTPConnection("127.0.0.1", 8081, timeout=1)
            try:
                connection.request("GET", "/health")
                response = connection.getresponse()
                assert response.status == 200
                assert json.loads(response.read()) == {"status": "ok"}
                break
            except OSError:
                time.sleep(0.2)
            finally:
                connection.close()
        else:
            raise AssertionError("Camera HTTP server did not become healthy")
    finally:
        server.terminate()
        server.wait(timeout=10)


def main():
    mode = sys.argv[1]
    assert mode in {"engine", "camera"}
    assert os.geteuid() != 0, "Runtime must not run as root"
    assert sys.prefix == "/opt/venv"
    assert sys.prefix != sys.base_prefix
    for command in ("git", "uv", "uvx", "pip", "pip3", "gcc", "g++", "make", "curl"):
        assert shutil.which(command) is None, command
    assert importlib.util.find_spec("pip") is None
    assert importlib.util.find_spec("setuptools") is None
    installed = check_locked_packages(mode)

    # Check native loading and real JPEG encoding/decoding, not just metadata.
    assert "CPUExecutionProvider" in onnxruntime.get_available_providers()
    _ = ncnn.Net()
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    encoded, data = cv2.imencode(".jpg", image)
    assert encoded
    assert Image.open(BytesIO(data.tobytes())).size == (32, 24)
    cache = Path(os.environ["XDG_CACHE_HOME"])
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "container-check").write_text("writable")
    Path("/usr/src/app/data/heartbeat").write_text("alive")
    try:
        Path("/opt/venv/.write-probe").touch()
    except OSError as error:
        if error.errno not in {errno.EACCES, errno.EROFS}:
            raise
    else:
        raise AssertionError("Application user can modify runtime dependencies")
    if mode == "engine":
        from pyroengine import SystemController

        assert SystemController
        assert not shutil.which("ffmpeg")
        subprocess.run([sys.executable, "run.py", "--help"], check=True, stdout=subprocess.DEVNULL)
    else:
        check_camera()
    print(json.dumps({"image": mode, "uid": os.geteuid(), "packages": len(installed), "status": "passed"}))


if __name__ == "__main__":
    main()
