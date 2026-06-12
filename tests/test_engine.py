import io
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from dotenv import load_dotenv
from PIL import Image

from pyroengine.engine import ContextCrop, Engine


def test_engine_offline(tmpdir_factory, mock_wildfire_image, mock_forest_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    engine = Engine(cache_folder=folder)

    # Cache saving
    ts = datetime.now().isoformat()
    engine._stage_alert(mock_wildfire_image, 0, datetime.now().isoformat(), bboxes="dummy")
    assert len(engine._alerts) == 1
    assert engine._alerts[0]["ts"] < datetime.now().isoformat()
    assert ts < engine._alerts[0]["ts"]
    assert engine._alerts[0]["media_id"] is None
    assert engine._alerts[0]["alert_id"] is None

    # inference
    engine = Engine(nb_consecutive_frames=4, cache_folder=folder, save_captured_frames=True)
    out = engine.predict(mock_forest_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 1
    assert engine._states["-1"]["ongoing"] is False
    # No raw preds on the forest image: nothing is kept in RAM for that frame
    assert engine._states["-1"]["last_predictions"][0][0] is None
    assert engine._states["-1"]["last_predictions"][0][1].shape[0] == 0
    assert engine._states["-1"]["last_predictions"][0][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][0][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][0][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 2
    assert engine._states["-1"]["ongoing"] is False
    assert engine._states["-1"]["last_predictions"][0][0] is None
    # Wildfire frame has raw preds: a compact context crop is kept instead of the full frame
    assert isinstance(engine._states["-1"]["last_predictions"][1][0], ContextCrop)
    assert engine._states["-1"]["last_predictions"][1][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][1][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][1][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][1][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 3
    assert engine._states["-1"]["ongoing"]
    assert engine._states["-1"]["last_predictions"][0][0] is None
    assert isinstance(engine._states["-1"]["last_predictions"][2][0], ContextCrop)
    assert engine._states["-1"]["last_predictions"][2][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][2][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][2][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][2][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float)
    assert 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 4
    assert engine._states["-1"]["ongoing"]
    assert engine._states["-1"]["last_predictions"][0][0] is None
    assert isinstance(engine._states["-1"]["last_predictions"][-1][0], ContextCrop)
    assert engine._states["-1"]["last_predictions"][-1][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][-1][1].shape[1] == 5
    assert len(engine._states["-1"]["last_predictions"][-1][2][0]) == 5
    assert engine._states["-1"]["last_predictions"][-1][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][-1][4] is False


def create_dummy_onnx_model(model_path):
    """Creates a small dummy ONNX model."""
    x = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])
    y = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])

    node = onnx.helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = onnx.helper.make_graph([node], "dummy_model", [x], [y])

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 10)])
    model.ir_version = 10

    onnx.save(model, model_path)


@pytest.fixture
def dummy_onnx_file():
    """Fixture to create a temporary ONNX file."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
        create_dummy_onnx_model(tmpfile.name)
        yield tmpfile.name  # returns file path


def test_valid_model_path(dummy_onnx_file):
    """Tests Engine instanciation with a valid input model_path"""
    instance = Engine(model_path=dummy_onnx_file)
    assert instance.model.format == "onnx"


@pytest.fixture
def invalid_onnx_file():
    """Fixture to create a temporary invalid ONNX file."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmpfile:
        Path(tmpfile.name).write_bytes(b"Invalid content")
        yield tmpfile.name  # returns file path


def test_invalid_model_content(invalid_onnx_file):
    """Tests Engine instantiation with an invalid ONNX model content."""
    with pytest.raises(RuntimeError, match="Failed to load the ONNX model"):
        # Engine instantiation with an invalid model : Classifier instnaciation should raise an error
        Engine(model_path=invalid_onnx_file)


# mock_isfile is a mock of the os.path.isfile() function which allows to simulate file existence
@patch("os.path.isfile")
def test_nonexistent_model(mock_isfile):
    """Tests Engine instanciation with a non-existent input model_path"""
    mock_isfile.return_value = False  # Simulates file non-existence
    with pytest.raises(ValueError, match=r"Model file not found: .*"):
        Engine(model_path="nonexistent.onnx")


@patch("os.path.isfile", return_value=True)
@patch("pathlib.Path.is_file", return_value=True)
def test_invalid_extension(mock_path_is_file, mock_os_isfile):
    """Tests Engine instanciation with a file format different than .onnx"""
    with pytest.raises(
        ValueError,
        match=r"Input model_path should point to an ONNX export but currently is",
    ):
        Engine(model_path="model.ncnn")


def test_engine_online(tmpdir_factory, mock_wildfire_stream, mock_wildfire_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # With API
    load_dotenv(Path(__file__).parent.parent.joinpath(".env").absolute())
    api_url = os.environ.get("API_URL")
    cam_creds = {"dummy_cam": (os.environ.get("API_TOKEN"), 66)}
    # Skip the API-related tests if the URL is not specified

    if isinstance(api_url, str):
        engine = Engine(
            api_url=api_url,
            conf_thresh=0.01,
            cam_creds=cam_creds,
            nb_consecutive_frames=4,
            frame_saving_period=3,
            cache_folder=folder,
        )
        # Heartbeat
        start_ts = datetime.now(timezone.utc).isoformat()
        response = engine.heartbeat("dummy_cam")
        assert response.status_code // 100 == 2
        json_respone = response.json()
        time.sleep(0.1)
        ts = datetime.now(timezone.utc).isoformat()

        assert start_ts < json_respone["last_active_at"] < ts
        # Send an alert
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 1
        assert len(engine._alerts) == 0
        assert engine._states["dummy_cam"]["ongoing"] is False

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 2

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 3

        assert engine._states["dummy_cam"]["ongoing"]
        # Check that a media and an alert have been registered
        engine._process_alerts()
        assert len(engine._alerts) == 0


@pytest.mark.parametrize(("save_detections_frames", "expected_backup_calls"), [(True, 1), (False, 0)])
def test_process_alerts_respects_save_detections_flag(tmp_path, save_detections_frames, expected_backup_calls):
    api_url = os.environ.get("API_URL")
    api_token = os.environ.get("API_TOKEN")

    if not api_url or not api_token:
        pytest.skip("API_URL and API_TOKEN must be set to run this test against the real API")

    cam_creds = {"dummy_cam": (api_token, 0)}
    engine = Engine(
        api_url=api_url,
        cache_folder=str(tmp_path),
        cam_creds=cam_creds,
        save_detections_frames=save_detections_frames,
    )

    # Provide a non-empty bbox list so the API accepts the payload
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="JPEG")
    engine._stage_alert(
        None,
        "dummy_cam",
        int(time.time()),
        bboxes=[(0.1, 0.1, 0.2, 0.2, 0.9)],
        jpeg_bytes=buf.getvalue(),
    )

    with patch.object(engine, "_local_backup") as mock_backup:
        engine._process_alerts()

    assert mock_backup.call_count == expected_backup_calls
    if len(engine._alerts) > 0:
        pytest.skip("Detection upload failed, alert left in cache")
    assert len(engine._alerts) == 0


def test_fill_empty_bboxes(tmp_path):
    """fill_empty_bboxes stamps a placeholder bbox at conf=0 on any empty alert
    and leaves non-empty alerts untouched."""
    engine = Engine(cache_folder=str(tmp_path))

    cam_id = "169.254.7.3_3"
    bboxes_seq = [
        [(0.436, 0.609, 0.44, 0.62, 0.089)],
        [(0.436, 0.609, 0.44, 0.62, 0.589)],
        [(0.436, 0.609, 0.44, 0.62, 0.489)],
        [],  # empty middle frame
        [(0.436, 0.609, 0.44, 0.62, 0.689)],
        [(0.436, 0.609, 0.44, 0.62, 0.389)],
    ]
    for i, bboxes in enumerate(bboxes_seq):
        engine._stage_alert(None, cam_id, i, bboxes=bboxes)

    engine.fill_empty_bboxes()

    assert all(alert["bboxes"] for alert in engine._alerts)
    # Previously-empty frame gets the placeholder bbox at conf=0
    assert engine._alerts[3]["bboxes"] == [(0.0, 0.0, 0.0001, 0.0001, 0.0)]
    # Non-empty frames untouched
    assert engine._alerts[0]["bboxes"][0][4] == pytest.approx(0.089)
    assert engine._alerts[5]["bboxes"][0][4] == pytest.approx(0.389)


def test_fill_empty_bboxes_all_empty_for_cam(tmp_path):
    """Even when every alert for a cam_id is empty, each one gets the placeholder."""
    engine = Engine(cache_folder=str(tmp_path))

    for i in range(3):
        engine._stage_alert(None, "169.254.7.3_3", i, bboxes=[])

    engine.fill_empty_bboxes()

    assert all(alert["bboxes"] == [(0.0, 0.0, 0.0001, 0.0001, 0.0)] for alert in engine._alerts)


def test_build_context_crop(tmp_path):
    """_build_context_crop keeps a compact JPEG region covering all raw preds, or None without preds."""
    engine = Engine(cache_folder=str(tmp_path))
    frame = Image.new("RGB", (2560, 1440))

    assert engine._build_context_crop(frame, np.empty((0, 5))) is None

    preds = np.array([[0.4, 0.4, 0.45, 0.45, 0.8]])
    context = engine._build_context_crop(frame, preds)
    assert isinstance(context, ContextCrop)
    assert (context.full_w, context.full_h) == (2560, 1440)
    region = Image.open(io.BytesIO(context.jpeg))
    # Region respects the 1024px floor and contains the pred area
    assert min(region.size) >= 1024
    assert context.left <= 0.4 * 2560
    assert context.left + region.size[0] >= 0.45 * 2560
    assert context.top <= 0.4 * 1440
    assert context.top + region.size[1] >= 0.45 * 1440
    # The point of the change: the stored payload is much smaller than the decoded frame
    assert len(context.jpeg) < 2560 * 1440 * 3 / 10

    # Elongated bbox: both axes are sized on the largest union side so the square
    # detection crop computed later (1.2x that side) always fits inside the region
    frame = Image.new("RGB", (3840, 2160))
    preds = np.array([[0.10, 0.45, 0.49, 0.50, 0.8]])  # ~1500x108 px
    context = engine._build_context_crop(frame, preds)
    region = Image.open(io.BytesIO(context.jpeg))
    crop_box = engine._compute_crop_box(preds.tolist(), 3840, 2160)
    assert region.size[1] >= crop_box[3] - crop_box[1]
    assert region.size[0] >= crop_box[2] - crop_box[0]

    # A large bbox is capped to the crop box plus a fixed margin, not blown up toward the
    # full frame, so RAM stays bounded. The crop box still fits inside the region.
    preds = np.array([[0.30, 0.30, 0.69, 0.69, 0.8]])  # ~1500x842 px square-ish, large
    context = engine._build_context_crop(frame, preds)
    region = Image.open(io.BytesIO(context.jpeg))
    crop_box = engine._compute_crop_box(preds.tolist(), 3840, 2160)
    assert region.size[0] < 3840  # not the full frame width
    assert region.size[0] >= crop_box[2] - crop_box[0]
    assert region.size[1] >= crop_box[3] - crop_box[1]


def test_cluster_bboxes():
    """Overlapping bboxes merge (transitively); distant ones stay separate."""
    a = (0.10, 0.10, 0.20, 0.20, 0.9)
    b = (0.15, 0.15, 0.25, 0.25, 0.8)  # overlaps a
    c = (0.24, 0.24, 0.30, 0.30, 0.7)  # overlaps b only -> same cluster via transitivity
    d = (0.80, 0.80, 0.90, 0.90, 0.6)  # far away

    clusters = Engine._cluster_bboxes([a, d, b, c])

    assert len(clusters) == 2
    sizes = sorted(len(members) for members in clusters)
    assert sizes == [1, 3]


def test_event_crop_boxes_frozen(tmp_path):
    """The crop box assigned to a jittering bbox stays identical across frames of one event."""
    engine = Engine(cache_folder=str(tmp_path))
    cam_key = "dummy_cam"
    engine._states[cam_key] = engine._new_state()
    full_w, full_h = 3840, 2160

    bbox_t0 = [0.40, 0.40, 0.45, 0.45, 0.8]
    engine._update_event_crop_boxes(cam_key, [bbox_t0], full_w, full_h)
    assert len(engine._states[cam_key]["event_crop_boxes"]) == 1
    box_t0 = engine._assign_crop_boxes([bbox_t0], cam_key, full_w, full_h)[0]

    # Slightly moved bbox on the next frame: same frozen box, no new one
    bbox_t1 = [0.41, 0.39, 0.46, 0.44, 0.7]
    engine._update_event_crop_boxes(cam_key, [bbox_t0, bbox_t1], full_w, full_h)
    box_t1 = engine._assign_crop_boxes([bbox_t1], cam_key, full_w, full_h)[0]
    assert box_t1 == box_t0
    assert len(engine._states[cam_key]["event_crop_boxes"]) == 1

    # A second detection far away gets its own frozen box, the first one is untouched
    bbox_far = [0.80, 0.80, 0.85, 0.85, 0.6]
    engine._update_event_crop_boxes(cam_key, [bbox_t1, bbox_far], full_w, full_h)
    assert len(engine._states[cam_key]["event_crop_boxes"]) == 2
    boxes = engine._assign_crop_boxes([bbox_t1, bbox_far], cam_key, full_w, full_h)
    assert boxes[0] == box_t0
    assert boxes[1] != box_t0

    # A plume that outgrew its frozen box re-anchors on a new, larger box
    bbox_grown = [0.35, 0.35, 0.50, 0.50, 0.9]
    engine._update_event_crop_boxes(cam_key, [bbox_grown, bbox_far], full_w, full_h)
    assert len(engine._states[cam_key]["event_crop_boxes"]) == 3
    box_grown = engine._assign_crop_boxes([bbox_grown], cam_key, full_w, full_h)[0]
    assert box_grown != box_t0
    assert box_grown[2] - box_grown[0] > box_t0[2] - box_t0[0]


def test_encode_detection_crops_one_per_bbox(tmp_path):
    """_encode_detection_crops returns one 224x224 JPEG per bbox, aligned by index."""
    engine = Engine(cache_folder=str(tmp_path))

    frame = Image.new("RGB", (1280, 720))
    # Paint the first bbox region red so the two crops have distinct content
    frame.paste((255, 0, 0), (0, 0, 250, 250))
    bboxes = [
        (0.05, 0.05, 0.15, 0.15, 0.9),
        (0.8, 0.7, 0.95, 0.9, 0.5),
    ]
    context = engine._build_context_crop(frame, np.array([list(b) for b in bboxes]))
    crop_boxes = [engine._compute_crop_box([b], 1280, 720) for b in bboxes]

    crops = engine._encode_detection_crops(context, bboxes, crop_boxes)

    assert crops is not None
    assert len(crops) == len(bboxes)
    for crop_bytes in crops:
        crop = Image.open(io.BytesIO(crop_bytes))
        assert crop.format == "JPEG"
        assert crop.size == (224, 224)
    # Distant bboxes must yield different crops, not one shared global crop
    assert crops[0] != crops[1]

    # First crop covers the red-painted region
    first = Image.open(io.BytesIO(crops[0])).convert("RGB")
    r, g, b = first.getpixel((112, 112))
    assert r > 150
    assert g < 100
    assert b < 100

    assert engine._encode_detection_crops(context, [], None) is None
    assert engine._encode_detection_crops(None, bboxes, crop_boxes) is None


def _build_engine_with_pose_stub(tmp_path, init_clock):
    """Build an Engine with the api_client stubbed and datetime.now() pinned to init_clock."""
    cam_id = "169.254.7.3_3"
    cam_creds = {cam_id: ("dummy_token", 7)}

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return init_clock if tz is None else init_clock.replace(tzinfo=tz)

    fake_client = MagicMock()
    fake_client.update_pose_image.return_value = MagicMock(text="ok")
    fake_client.update_last_image.return_value = MagicMock(text="ok")
    fake_client.list_pose_masks.return_value = MagicMock(
        raise_for_status=MagicMock(),
        json=MagicMock(return_value=[]),
    )
    fake_client.heartbeat.return_value = MagicMock()

    with (
        patch("pyroengine.engine.datetime", _FrozenDateTime),
        patch("pyroengine.engine.client.Client", return_value=fake_client),
    ):
        engine = Engine(api_url="http://stub", cache_folder=str(tmp_path), cam_creds=cam_creds)

    return engine, fake_client, cam_id


def _run_predict_at(engine, cam_id, image, run_clock):
    class _RunDateTime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:
            return run_clock if tz is None else run_clock.replace(tzinfo=tz)

    with patch("pyroengine.engine.datetime", _RunDateTime):
        engine.predict(image, cam_id)


def test_pose_image_skipped_when_engine_starts_after_noon(tmp_path, mock_forest_image):
    init_clock = datetime(2026, 5, 1, 14, 0, 0)
    engine, fake_client, cam_id = _build_engine_with_pose_stub(tmp_path, init_clock)

    _run_predict_at(engine, cam_id, mock_forest_image, init_clock + timedelta(seconds=1))

    fake_client.update_pose_image.assert_not_called()


def test_pose_image_sent_at_noon_crossing(tmp_path, mock_forest_image):
    init_clock = datetime(2026, 5, 1, 11, 30, 0)
    engine, fake_client, cam_id = _build_engine_with_pose_stub(tmp_path, init_clock)

    # Before noon: no send.
    _run_predict_at(engine, cam_id, mock_forest_image, datetime(2026, 5, 1, 11, 59, 0))
    fake_client.update_pose_image.assert_not_called()

    # After noon: one send.
    _run_predict_at(engine, cam_id, mock_forest_image, datetime(2026, 5, 1, 12, 0, 5))
    assert fake_client.update_pose_image.call_count == 1

    # Same day, no resend.
    _run_predict_at(engine, cam_id, mock_forest_image, datetime(2026, 5, 1, 13, 0, 0))
    assert fake_client.update_pose_image.call_count == 1


def test_pose_image_sent_again_next_day(tmp_path, mock_forest_image):
    init_clock = datetime(2026, 5, 1, 9, 0, 0)
    engine, fake_client, cam_id = _build_engine_with_pose_stub(tmp_path, init_clock)

    _run_predict_at(engine, cam_id, mock_forest_image, datetime(2026, 5, 1, 12, 0, 5))
    assert fake_client.update_pose_image.call_count == 1

    # Next day at noon: another send.
    _run_predict_at(engine, cam_id, mock_forest_image, datetime(2026, 5, 2, 12, 0, 5))
    assert fake_client.update_pose_image.call_count == 2


def test_engine_occlusion(tmpdir_factory, mock_wildfire_stream, mock_wildfire_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # With API
    load_dotenv(Path(__file__).parent.parent.joinpath(".env").absolute())
    api_url = os.environ.get("API_URL")
    # Use pose 356 which has an occlusion mask "(0.0,0.5,0.05,0.65)" covering the model's prediction area
    cam_creds = {"dummy_cam": (os.environ.get("API_TOKEN"), 356)}
    # Skip the API-related tests if the URL is not specified

    if isinstance(api_url, str):
        engine = Engine(
            api_url=api_url,
            conf_thresh=0.01,
            cam_creds=cam_creds,
            nb_consecutive_frames=4,
            frame_saving_period=3,
            cache_folder=folder,
        )
        # Heartbeat
        start_ts = datetime.now(timezone.utc).isoformat()
        response = engine.heartbeat("dummy_cam")
        assert response.status_code // 100 == 2
        json_respone = response.json()
        time.sleep(0.1)
        ts = datetime.now(timezone.utc).isoformat()

        assert start_ts < json_respone["last_active_at"] < ts

        # First predict triggers the occlusion mask fetch from the API
        engine.predict(mock_wildfire_image, "dummy_cam")
        # Verify masks were fetched and parsed correctly
        assert "dummy_cam" in engine.occlusion_masks
        assert len(engine.occlusion_masks["dummy_cam"]) > 0

        assert len(engine._states["dummy_cam"]["last_predictions"]) == 1
        assert len(engine._alerts) == 0
        assert engine._states["dummy_cam"]["ongoing"] is False

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 2

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 3

        # Predictions are filtered by the occlusion mask, so no wildfire is detected
        assert engine._states["dummy_cam"]["ongoing"] is False
