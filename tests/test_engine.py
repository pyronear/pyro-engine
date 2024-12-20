import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv
from PIL import Image

from pyroengine.engine import Engine
from pyroengine.sensors import ReolinkCamera


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
        f"{api_url}/api/v1/login/creds",
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
    api_url = os.environ.get("API_URL")
    superuser_login = os.environ.get("API_LOGIN")
    superuser_pwd = os.environ.get("API_PWD")

    camera_id = 7  # created in the dev env

    # Skip the API-related tests if the URL is not specified
    if isinstance(api_url, str):

        superuser_auth = {
            "Authorization": f"Bearer {get_token(api_url, superuser_login, superuser_pwd)}",
            "Content-Type": "application/json",
        }

        token = requests.post(f"{api_url}/api/v1/cameras/{camera_id}/token", headers=superuser_auth).json()[
            "access_token"
        ]

        splitted_cam_creds = {
            "dummy_cam": f"{ token }",
        }

        engine = Engine(
            api_host=api_url,
            cam_creds=splitted_cam_creds,
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
        json_response = response.json()

        assert start_ts < json_response["last_active_at"] < ts
        # Send an alert
        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 1
        assert len(engine._alerts) == 0
        assert engine._states["dummy_cam"]["ongoing"] is False

        engine.predict(mock_wildfire_image, "dummy_cam")
        assert len(engine._states["dummy_cam"]["last_predictions"]) == 2

        assert engine._states["dummy_cam"]["ongoing"] is True
        # Check that a media and an alert have been registered
        CAM_USER = os.environ.get("CAM_USER")
        CAM_PWD = os.environ.get("CAM_PWD")
        engine._process_alerts([ReolinkCamera("dummy_cam", CAM_USER, CAM_PWD, cam_azimuths=[120])])
        assert len(engine._alerts) == 0
