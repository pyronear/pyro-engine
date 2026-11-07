# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Load ONNX anonymizer exports from fresh archives and existing caches."""

import io
import tarfile
from pathlib import Path

import pytest

from pyro_camera_api.services import vision


@pytest.mark.parametrize("filenames", [("best.onnx",), ("model.onnx",), ("best.onnx", "model.onnx")])
def test_onnx_archive_filenames(tmp_path, monkeypatch, filenames):
    archive = tmp_path / vision.MODEL_NAME.replace("ncnn", "onnx")
    with tarfile.open(archive, "w:gz") as tar:
        for filename in filenames:
            content = filename.encode()
            entry = tarfile.TarInfo(filename)
            entry.size = len(content)
            tar.addfile(entry, io.BytesIO(content))

    # Opening the file stands in for native session creation: a wrong filename fails.
    monkeypatch.setattr(vision.onnxruntime, "InferenceSession", lambda path: Path(path).read_bytes())
    for _ in range(2):  # First extract the archive, then reuse the extracted cache.
        anonymizer = vision.Anonymizer(model_folder=str(tmp_path), model_format="onnx")
        assert anonymizer.ort_session == filenames[0].encode()


def test_explicit_onnx_path(tmp_path, monkeypatch):
    model = tmp_path / "custom.onnx"
    model.write_bytes(b"custom model")
    monkeypatch.setattr(vision.onnxruntime, "InferenceSession", lambda path: Path(path).read_bytes())
    anonymizer = vision.Anonymizer(model_path=str(model))
    assert anonymizer.ort_session == b"custom model"
