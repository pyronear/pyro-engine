# Copyright (C) 2022-2026, Pyronear.
# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.
"""Accuracy guards against the original preprocessing and suppression behavior."""

import cv2
import numpy as np
import pytest
from PIL import Image
from pyro_predictor import Classifier
from pyro_predictor.utils import box_iou, letterbox, nms


def reference_letterbox(im, new_shape=(1024, 1024), color=(114, 114, 114), auto=False, stride=32):
    im = np.array(im)
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    h, w = im.shape[:2]
    padded = np.zeros((h + top + bottom, w + left + right, 3)) + color
    padded[top : top + h, left : left + w, :] = im
    return padded.astype("uint8"), (left, top)


def reference_box_iou(box1, box2, eps=1e-7):
    (a1, a2), (b1, b2) = np.split(box1, 2, 1), np.split(box2, 2, 1)
    inter = (np.minimum(a2, b2[:, None, :]) - np.maximum(a1, b1[:, None, :])).clip(0).prod(2)
    return inter / ((a2 - a1).prod(1) + (b2 - b1).prod(1)[:, None] - inter + eps)


def reference_nms(boxes, threshold=0):
    boxes = boxes[boxes[:, -1].argsort()]
    if not len(boxes):
        return []
    indices = np.arange(len(boxes))
    overlaps = reference_box_iou(boxes[:, :4], boxes[:, :4])
    for i in range(len(boxes)):
        others = indices[indices != i]
        if np.any(overlaps[i, others] > threshold):
            indices = indices[indices != i]
    return boxes[indices]


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16, np.int64, np.uint8])
@pytest.mark.parametrize("counts", [(0, 3), (3, 0), (1, 1), (13, 29)])
def test_box_iou_preserves_values_and_orientation(dtype, counts):
    rng = np.random.default_rng(42)
    a = (rng.random((counts[0], 4)) * 100).astype(dtype)
    b = (rng.random((counts[1], 4)) * 100).astype(dtype)
    np.testing.assert_array_equal(box_iou(a, b), reference_box_iou(a, b))
    assert box_iou(a, b).shape == (counts[1], counts[0])


@pytest.mark.parametrize("shape", [(1, 3), (64, 64), (721, 1280), (1081, 1920), (2160, 3840)])
@pytest.mark.parametrize("auto", [True, False])
def test_letterbox_preserves_pixels_and_input(shape, auto):
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    before = image.copy()
    for color in [(114, 114, 114), (12.9, 260, -2.5)]:
        expected, expected_pad = reference_letterbox(image, (129, 256), color, auto)
        actual, actual_pad = letterbox(image, (129, 256), color, auto)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, before)
        assert actual_pad == expected_pad
        assert actual.dtype == np.uint8


def test_letterbox_noncontiguous_input():
    image = np.arange(90 * 60 * 3, dtype=np.uint8).reshape(90, 60, 3)[::2, ::2]
    expected, pad = reference_letterbox(image, 128)
    actual, actual_pad = letterbox(image, 128)
    np.testing.assert_array_equal(actual, expected)
    assert actual_pad == pad


@pytest.mark.parametrize("shape", [(1024, 1024), (721, 1280), (1081, 1920), (2160, 3840)])
def test_onnx_preprocessing_is_bitwise_identical(shape):
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8))
    padded, expected_pad = reference_letterbox(np.array(image), 1024)
    expected = np.ascontiguousarray(np.expand_dims(padded.astype("float32"), 0).transpose(0, 3, 1, 2))
    expected /= 255.0
    model = Classifier.__new__(Classifier)
    model.imgsz, model.format = 1024, "onnx"
    actual, pad = model.prep_process(image)
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert actual.dtype == np.float32
    assert pad == expected_pad


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("threshold", [-1, 0, 0.1, 0.5, 1])
@pytest.mark.parametrize("count", [0, 1, 16, 129, 384])
def test_nms_preserves_original_suppression(dtype, threshold, count):
    rng = np.random.default_rng(42)
    for dense in [False, True]:
        xy = rng.uniform(0, 1 if not dense else 0.1, (count, 2))
        wh = rng.uniform(0, 0.1 if not dense else 0.8, (count, 2))
        # Rounded confidences exercise tie ordering.
        boxes = np.column_stack((xy, xy + wh, np.round(rng.random(count), 1))).astype(dtype)
        before = boxes.copy()
        np.testing.assert_array_equal(nms(boxes, threshold), reference_nms(boxes, threshold))
        np.testing.assert_array_equal(boxes, before)


def test_nms_overlap_chain_keeps_original_behavior():
    # A overlaps B and B overlaps C; A does not overlap C. The original drops
    # both A and B, unlike greedy NMS, which would keep A and C.
    boxes = np.array([[0, 0, 2, 2, 0.1], [1, 0, 3, 2, 0.2], [2, 0, 4, 2, 0.3]])
    np.testing.assert_array_equal(nms(boxes), boxes[-1:])


def test_nms_handles_touching_degenerate_and_nonfinite_boxes():
    boxes = np.array([
        [0, 0, 0, 1, 0.1],
        [0, 0, 1, 1, 0.2],
        [1, 0, 2, 1, 0.3],
        [0, 0, np.nan, 1, 0.4],
        [2, 2, 1, 1, 0.5],
    ])
    for threshold in [0, 0.1, float("nan")]:
        np.testing.assert_array_equal(nms(boxes, threshold), reference_nms(boxes, threshold))


def test_onnx_session_options_are_forwarded(tmp_path, monkeypatch):
    from unittest.mock import MagicMock

    import onnxruntime
    from pyro_predictor import vision

    path = tmp_path / "model.onnx"
    path.touch()
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 2
    factory = MagicMock()
    monkeypatch.setattr(vision.onnxruntime, "InferenceSession", factory)
    Classifier(model_path=str(path), onnx_session_options=options, verbose=False)
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
