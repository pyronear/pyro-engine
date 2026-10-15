# Copyright (C) 2022-2026, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
"""Accuracy guards against the original preprocessing and suppression behavior."""

import cv2
import numpy as np
import pytest
from PIL import Image
from pyro_predictor.utils import letterbox, nms


def test_letterbox_preserves_pixels_and_input():
    image = np.arange(3 * 7 * 3, dtype=np.uint8).reshape(3, 7, 3)
    before = image.copy()
    expected = np.full((10, 12, 3), 114, dtype=np.uint8)
    expected[2:7] = cv2.resize(image, (12, 5), interpolation=cv2.INTER_LINEAR)
    actual, pad = letterbox(image, (10, 12))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, before)
    assert pad == (0, 2)
    assert actual.dtype == np.uint8


def test_nms_overlap_chain_keeps_original_behavior():
    # A overlaps B and B overlaps C; A does not overlap C. The original drops
    # both A and B, unlike greedy NMS, which would keep A and C.
    boxes = np.array([[0, 0, 2, 2, 0.1], [1, 0, 3, 2, 0.2], [2, 0, 4, 2, 0.3]])
    np.testing.assert_array_equal(nms(boxes), boxes[-1:])


def test_engine_forwards_onnx_session_options(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import onnxruntime
    from pyro_predictor import vision

    from pyroengine.engine import Engine

    path = tmp_path / "model.onnx"
    path.touch()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    factory = MagicMock()
    monkeypatch.setattr(vision.onnxruntime, "InferenceSession", factory)
    Engine(model_path=str(path), cache_folder=str(tmp_path), onnx_session_options=options, verbose=False)
    assert factory.call_args.kwargs["sess_options"] is options


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P", "CMYK"])
@pytest.mark.parametrize("use_engine", [False, True])
def test_model_receives_identical_rgb_pixels(tmp_path, monkeypatch, mode, use_engine):
    from unittest.mock import MagicMock

    from pyro_predictor import Predictor
    from pyro_predictor import predictor as predictor_module

    from pyroengine.engine import Engine

    model = MagicMock(return_value=np.empty((0, 5)))
    monkeypatch.setattr(predictor_module, "Classifier", lambda **_kwargs: model)
    instance = Engine(cache_folder=str(tmp_path), verbose=False) if use_engine else Predictor(verbose=False)
    frame = Image.fromarray(np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)).convert(mode)
    # Engine's existing JPEG upload path does not accept RGBA/P images; isolate
    # model input conversion from that independent encoding constraint.
    if use_engine and mode in ("RGBA", "P"):
        monkeypatch.setattr(frame, "save", lambda *_args, **_kwargs: None)
    instance.predict(frame)
    actual = model.call_args.args[0]
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(frame.convert("RGB")))
    assert actual.mode == "RGB"
    if mode == "RGB":
        assert actual is frame
