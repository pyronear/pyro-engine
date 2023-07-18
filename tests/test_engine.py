import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from pyroengine.engine import Engine


def test_engine_offline(tmpdir_factory, mock_wildfire_image, mock_forest_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    engine = Engine(cache_folder=folder)

    # Cache saving
    _ts = datetime.utcnow().isoformat()
    engine._stage_alert(mock_wildfire_image, 0, localization="dummy")
    assert len(engine._alerts) == 1
    assert engine._alerts[0]["ts"] < datetime.utcnow().isoformat() and _ts < engine._alerts[0]["ts"]
    assert engine._alerts[0]["media_id"] is None
    assert engine._alerts[0]["alert_id"] is None

    # Cache dump
    engine._dump_cache()
    assert engine._cache.joinpath("pending_alerts.json").is_file()
    with open(engine._cache.joinpath("pending_alerts.json"), "rb") as f:
        cache_dump = json.load(f)
    assert isinstance(cache_dump, list) and len(cache_dump) == 1 and len(engine._alerts) == 1
    assert cache_dump[0] == {
        "frame_path": str(engine._cache.joinpath("pending_frame0.jpg")),
        "cam_id": 0,
        "ts": engine._alerts[0]["ts"],
    }
    # Overrites cache files
    engine._dump_cache()

    # Cache dump loading
    engine = Engine(cache_folder=folder + "model.onnx")
    assert len(engine._alerts) == 1
    engine.clear_cache()

    # inference
    engine = Engine(alert_relaxation=3, cache_folder=folder + "model.onnx")
    out = engine.predict(mock_forest_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert engine._states["-1"]["consec"] == 0
    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert engine._states["-1"]["consec"] == 1
    # Alert relaxation
    assert not engine._states["-1"]["ongoing"]
    out = engine.predict(mock_wildfire_image)
    assert engine._states["-1"]["consec"] == 2
    out = engine.predict(mock_wildfire_image)
    assert engine._states["-1"]["consec"] == 3
    assert engine._states["-1"]["ongoing"]


def test_engine_online(tmpdir_factory, mock_wildfire_stream, mock_wildfire_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # With API
    load_dotenv(Path(__file__).parent.parent.joinpath(".env").absolute())
    api_url = os.environ.get("API_URL")
    lat = os.environ.get("LAT")
    lon = os.environ.get("LON")
    cam_creds = {"dummy_cam": {"login": os.environ.get("API_LOGIN"), "password": os.environ.get("API_PWD")}}
    # Skip the API-related tests if the URL is not specified
    if isinstance(api_url, str):
        engine = Engine(
            folder + "model.onnx",
            api_url=api_url,
            cam_creds=cam_creds,
            latitude=float(lat),
            longitude=float(lon),
            alert_relaxation=2,
            frame_saving_period=3,
            cache_folder=folder,
            frame_size=(256, 384),
        )
        # Heartbeat
        start_ts = datetime.utcnow().isoformat()
        response = engine.heartbeat("dummy_cam")
        assert response.status_code // 100 == 2
        ts = datetime.utcnow().isoformat()
        json_respone = response.json()
        assert start_ts < json_respone["last_ping"] < ts
        # Send an alert
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._alerts) == 0 and engine._states["dummy_cam"]["consec"] == 1
        assert engine._states["dummy_cam"]["frame_count"] == 1
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert engine._states["dummy_cam"]["consec"] == 2 and engine._states["dummy_cam"]["ongoing"]
        assert engine._states["dummy_cam"]["frame_count"] == 2
        # Check that a media and an alert have been registered
        assert len(engine._alerts) == 0
        # Upload a frame
        response = engine._upload_frame("dummy_cam", mock_wildfire_stream)
        assert response.status_code // 100 == 2
        # Upload frame in process
        engine.predict(mock_wildfire_image, "dummy_cam")
        # Check that a new media has been created & uploaded
        assert engine._states["dummy_cam"]["frame_count"] == 0
