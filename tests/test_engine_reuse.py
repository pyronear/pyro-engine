import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pyroengine.engine import Engine


@pytest.fixture
def engine_factory(monkeypatch, tmp_path):
    def classifier(**kwargs):
        model = MagicMock(format="ncnn", imgsz=1024, conf=kwargs["conf"], iou=0, max_bbox_size=0.4)
        model.model = object()

        def predict(frame, masks):
            score = frame.getpixel((0, 0))[0] / 255
            return (
                np.array([[0.1, 0.2, 0.3, 0.4, score]], dtype=np.float32)
                if score > model.conf and not masks
                else np.empty((0, 5), dtype=np.float32)
            )

        model.side_effect = predict
        return model

    monkeypatch.setattr("pyro_predictor.predictor.Classifier", classifier)
    return lambda reuse=False: Engine(
        cache_folder=str(tmp_path), nb_consecutive_frames=4, reuse_identical_frames=reuse, verbose=False
    )


def test_reuse_preserves_per_camera_alert_history(engine_factory):
    baseline, cached = engine_factory(), engine_factory(True)
    fire, forest = Image.new("RGB", (32, 32), "red"), Image.new("RGB", (32, 32))
    for cam, frame in [("a", fire), ("b", forest)] * 4 + [("a", forest)] * 4:
        assert baseline.predict(frame.copy(), cam) == cached.predict(frame.copy(), cam)
        a, b = baseline._states[cam], cached._states[cam]
        assert a["ongoing"] == b["ongoing"]
        assert len(a["last_predictions"]) == len(b["last_predictions"])
        for left, right in zip(a["last_predictions"], b["last_predictions"]):
            np.testing.assert_array_equal(left[1], right[1])
            assert left[2] == right[2]
            assert left[5] == right[5]
        if len(b["last_predictions"]) > 1:
            assert not np.shares_memory(b["last_predictions"][-1][1], b["last_predictions"][-2][1])
    assert baseline.model.call_count == 12
    assert cached.model.call_count == 3


@pytest.mark.parametrize("change", ["pixels", "size", "mask", "threshold", "backend"])
def test_reuse_invalidates_changed_inputs(engine_factory, change):
    engine = engine_factory(True)
    frame = Image.new("RGB", (32, 32))
    masks = {"area": [0.0, 0.0, 0.1, 0.1]}
    engine.occlusion_masks["a"] = masks
    engine.predict(frame, "a")
    engine.predict(frame.copy(), "a")
    assert engine.model.call_count == 1
    if change == "pixels":
        before, after = io.BytesIO(), io.BytesIO()
        frame.save(before, "JPEG", quality=80)
        frame.putpixel((0, 0), (1, 0, 0))
        frame.save(after, "JPEG", quality=80)
        assert before.getvalue() == after.getvalue()
    elif change == "size":
        frame = Image.new("RGB", (64, 16))
    elif change == "mask":
        masks["area"][0] = 0.05
    elif change == "threshold":
        engine.model.conf = 0.9
    else:
        engine.model.model = object()
    engine.predict(frame, "a")
    assert engine.model.call_count == 2


def test_fake_predictions_do_not_seed_reuse(engine_factory):
    engine = engine_factory(True)
    frame = Image.new("RGB", (32, 32), "red")
    engine.predict(frame)
    engine.predict(frame, fake_pred=np.empty((0, 5)))
    engine.predict(frame)
    assert engine.model.call_count == 2


def test_failed_prediction_does_not_seed_reuse(engine_factory):
    engine = engine_factory(True)
    engine.predict(Image.new("RGB", (32, 32)))
    frame = Image.new("RGB", (32, 32), "red")
    with patch.object(engine, "_build_context_crop", side_effect=RuntimeError("crop failed")):
        with pytest.raises(RuntimeError, match="crop failed"):
            engine.predict(frame)
    engine.predict(frame)
    assert engine.model.call_count == 3
