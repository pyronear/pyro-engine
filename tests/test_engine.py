import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

from pyroengine.engine import Engine


def test_engine_offline(tmpdir_factory, mock_wildfire_image, mock_forest_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    engine = Engine(cache_folder=folder)

    # Cache saving
    _ts = datetime.now().isoformat()
    engine._stage_alert(mock_wildfire_image, 0, None, datetime.now().isoformat(), bboxes="dummy")
    assert len(engine._alerts) == 1
    assert engine._alerts[0]["ts"] < datetime.now().isoformat() and _ts < engine._alerts[0]["ts"]

    # Cache dump
    engine._dump_cache()
    assert engine._cache.joinpath("pending_alerts.json").is_file()
    with open(engine._cache.joinpath("pending_alerts.json"), "rb") as f:
        cache_dump = json.load(f)
    assert isinstance(cache_dump, list) and len(cache_dump) == 1 and len(engine._alerts) == 1
    assert cache_dump[0] == {
        "frame_path": str(engine._cache.joinpath("pending_frame0.jpg")),
        "cam_id": 0,
        "pose_id": None,
        "ts": engine._alerts[0]["ts"],
        "bboxes": "dummy",
    }
    # Overrites cache files
    engine._dump_cache()

    # Cache dump loading
    engine = Engine(cache_folder=folder)
    assert len(engine._alerts) == 1
    engine.clear_cache()

    # inference
    engine = Engine(nb_consecutive_frames=4, cache_folder=folder, save_captured_frames=True)
    out = engine.predict(mock_forest_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 1
    assert engine._states["-1"]["ongoing"] is False
    assert isinstance(engine._states["-1"]["last_predictions"][0][0], Image.Image)
    assert engine._states["-1"]["last_predictions"][0][1].shape[0] == 0
    assert engine._states["-1"]["last_predictions"][0][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][0][2] == []
    assert engine._states["-1"]["last_predictions"][0][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][0][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 2
    assert engine._states["-1"]["ongoing"] is False
    assert isinstance(engine._states["-1"]["last_predictions"][0][0], Image.Image)
    assert engine._states["-1"]["last_predictions"][1][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][1][1].shape[1] == 5
    assert engine._states["-1"]["last_predictions"][1][2] == []
    assert engine._states["-1"]["last_predictions"][1][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][1][4] is False

    out = engine.predict(mock_wildfire_image)
    assert isinstance(out, float) and 0 <= out <= 1
    assert len(engine._states["-1"]["last_predictions"]) == 3
    assert engine._states["-1"]["ongoing"] is True
    assert isinstance(engine._states["-1"]["last_predictions"][0][0], Image.Image)
    assert engine._states["-1"]["last_predictions"][2][1].shape[0] > 0
    assert engine._states["-1"]["last_predictions"][2][1].shape[1] == 5
    assert len(engine._states["-1"]["last_predictions"][-1][2][0]) == 5
    assert engine._states["-1"]["last_predictions"][2][3] < datetime.now().isoformat()
    assert engine._states["-1"]["last_predictions"][2][4] is False


def get_token(api_url: str, login: str, pwd: str) -> str:
    response = requests.post(
        f"{api_url}/login/creds",
        data={"username": login, "password": pwd},
        timeout=5,
    )
    if response.status_code != 200:
        raise ValueError(response.json()["detail"])
    return response.json()["access_token"]


def test_engine_online(tmpdir_factory, mock_wildfire_stream, mock_wildfire_image):
    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))
    # With API
    load_dotenv(Path(__file__).parent.parent.joinpath(".env").absolute())
    api_host = os.environ.get("API_URL")
    camera_id = os.environ.get("CAMERA_ID")
    superuser_login = os.environ.get("SUPERADMIN_LOGIN")
    superuser_pwd = os.environ.get("SUPERADMIN_PWD")
    lat = os.environ.get("LAT")
    lon = os.environ.get("LON")

    # Skip the API-related tests if the URL is not specified
    if isinstance(api_host, str):
        api_url = api_host + "/api/v1"

        superuser_auth = {
            "Authorization": f"Bearer {get_token(api_url, superuser_login, superuser_pwd)}",
            "Content-Type": "application/json",
        }

        token = requests.post(f"{api_url}/cameras/{camera_id}/token", headers=superuser_auth)["access_token"]

        cam_creds = {
            "dummy_cam": {
                "brand": "reolink",
                "name": "cam-1",
                "type": "static",
                "token": token,
            }
        }

        engine = Engine(
            api_url=api_url,
            cam_creds=cam_creds,
            latitude=float(lat),
            longitude=float(lon),
            nb_consecutive_frames=4,
            frame_saving_period=3,
            cache_folder=folder,
            frame_size=(256, 384),
        )
        # Heartbeat
        start_ts = datetime.now(timezone.utc).isoformat()
        response = engine.heartbeat("dummy_cam")
        assert response.status_code // 100 == 2
        ts = datetime.now(timezone.utc).isoformat()
        json_respone = response.json()
        assert start_ts < json_respone["last_ping"] < ts
        # Send an alert
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 1
        assert len(engine._alerts) == 0
        assert engine._states["dummy_cam"]["ongoing"] is False

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 2

        assert engine._states["dummy_cam"]["ongoing"] is True
        # Check that a media and an alert have been registered
        engine._process_alerts()
        assert len(engine._alerts) == 0
