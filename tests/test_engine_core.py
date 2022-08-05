import json
from datetime import datetime

from pyroengine.core import Engine


def test_engine(tmpdir_factory, mock_classification_image):

    # Cache
    folder = str(tmpdir_factory.mktemp("engine_cache"))

    # No API
    engine = Engine("pyronear/rexnet1_3x", cache_folder=folder)

    # Cache saving
    _ts = datetime.utcnow().isoformat()
    engine._stage_alert(mock_classification_image, 0)
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

    # Cache dump loading
    engine = Engine("pyronear/rexnet1_3x", cache_folder=folder)
    assert len(engine._alerts) == 1
    engine.clear_cache()

    # inference
    engine = Engine("pyronear/rexnet1_3x", cache_folder=folder)
    out = engine.predict(mock_classification_image, 0)
    assert isinstance(out, float) and 0 <= out <= 1
    # Alert relaxation
    assert not engine._states["-1"]["ongoing"]
    out = engine.predict(mock_classification_image, 0)
    out = engine.predict(mock_classification_image, 0)
    assert engine._states["-1"]["ongoing"]

    # With API
