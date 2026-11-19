import importlib.util
import pathlib
from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image

from pyroengine.core import NIGHT_FRAMES_BEFORE_RESET

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "replay_folder_cam.py"
spec = importlib.util.spec_from_file_location("replay_folder_cam", SCRIPT)
assert spec is not None
assert spec.loader is not None
replay_folder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay_folder)

T0 = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
DAY = (200, 150, 100)
NIGHT = (120, 120, 120)  # infrared frames are grey


def write_frame(folder: pathlib.Path, captured_at: datetime, color=DAY) -> pathlib.Path:
    path = folder / f"{captured_at.strftime('%Y%m%dT%H%M%SZ')}.jpg"
    Image.new("RGB", (32, 32), color).save(path)
    return path


class FakeEngine:
    """predict() returns the scripted confidence; a positive frame opens an event and queues an alert."""

    conf_thresh = 0.25

    def __init__(self, confs) -> None:
        self.confs = list(confs)
        self._states = {}
        self._alerts = []
        self.predicted = []
        self.sent = []

    def predict(self, _frame, cam_id, captured_at=None):
        self.predicted.append(captured_at)
        conf = self.confs.pop(0) if self.confs else 0.0
        state = self._states.setdefault(cam_id, {"ongoing": False, "temporal_events": []})
        state["ongoing"] = conf > self.conf_thresh
        if state["ongoing"]:
            self._alerts.append({"ts": captured_at})
        return conf

    def _process_alerts(self):
        self.sent.extend(alert["ts"] for alert in self._alerts)
        self._alerts.clear()


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_replayer(engine, clock, **kwargs):
    params = {"interval": 0.0, "sequence_gap": 120.0, "sequence_pause": 150.0}
    params.update(kwargs)
    return replay_folder.Replayer(engine, "CAM", clock=clock, sleep=clock.sleep, **params)


def test_list_frames_filters_and_sorts(tmp_path):
    for minutes in (2, 0, 1, 3):
        write_frame(tmp_path, T0 + timedelta(minutes=minutes))
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "20261310T000000Z.jpg").write_text("bad date")
    frames = replay_folder.list_frames(tmp_path, since=T0, until=T0 + timedelta(minutes=3))
    assert [ts for ts, _ in frames] == [T0 + timedelta(minutes=1), T0 + timedelta(minutes=2)]


def test_progress_roundtrip(tmp_path):
    progress = tmp_path / "progress.json"
    assert replay_folder.load_progress(progress) is None
    replay_folder.save_progress(progress, T0)
    assert replay_folder.load_progress(progress) == T0


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 11, 10, 17, 30, tzinfo=timezone.utc), datetime(2026, 11, 11, 5, 0, tzinfo=timezone.utc)),
        (datetime(2026, 11, 10, 3, 0, tzinfo=timezone.utc), datetime(2026, 11, 10, 5, 0, tzinfo=timezone.utc)),
    ],
)
def test_next_occurrence(now, expected):
    assert replay_folder.next_occurrence("05:00", now) == expected


def test_pause_only_between_distant_detections(tmp_path):
    # Detection at 09:00, quiet frames 30 s apart, a frame 3 min later: one pause before it
    times = [T0, T0 + timedelta(seconds=30), T0 + timedelta(seconds=60), T0 + timedelta(minutes=3)]
    frames = [(ts, write_frame(tmp_path, ts)) for ts in times]
    engine, clock = FakeEngine([0.9, 0.0, 0.0, 0.9]), FakeClock()
    replayer = make_replayer(engine, clock)
    assert replayer.run(frames)
    assert clock.sleeps == [150.0]
    assert engine.sent == [T0.isoformat(), times[3].isoformat()]
    assert replayer.stats["pauses"] == 1


def test_pause_counts_time_already_spent(tmp_path):
    times = [T0, T0 + timedelta(minutes=5)]
    frames = [(ts, write_frame(tmp_path, ts)) for ts in times]
    engine, clock = FakeEngine([0.9, 0.0]), FakeClock()
    make_replayer(engine, clock, interval=100.0).run(frames)
    assert clock.sleeps == [100.0, 50.0, 100.0]


def test_night_frames_skipped_and_reset_state(tmp_path):
    times = [T0 + timedelta(minutes=i) for i in range(NIGHT_FRAMES_BEFORE_RESET + 1)]
    frames = [(times[0], write_frame(tmp_path, times[0]))]
    frames += [(ts, write_frame(tmp_path, ts, NIGHT)) for ts in times[1:]]
    engine, clock = FakeEngine([0.0]), FakeClock()
    replayer = make_replayer(engine, clock)
    replayer.run(frames)
    assert engine.predicted == [times[0].isoformat()]
    assert "CAM" not in engine._states
    assert replayer.stats["night"] == NIGHT_FRAMES_BEFORE_RESET


def test_limit_waits_for_event_end_and_saves_progress(tmp_path):
    times = [T0 + timedelta(seconds=30 * i) for i in range(5)]
    frames = [(ts, write_frame(tmp_path, ts)) for ts in times]
    progress = tmp_path / "progress.json"
    engine, clock = FakeEngine([0.0, 0.9, 0.9, 0.0, 0.0]), FakeClock()
    replayer = make_replayer(engine, clock, progress_file=progress)
    # The limit is reached on an ongoing event: the replay goes on until it ends
    assert not replayer.run(frames, limit=2)
    assert engine.predicted == [ts.isoformat() for ts in times[:4]]
    assert replay_folder.load_progress(progress) == times[3]


def test_stop_at(tmp_path):
    frames = [(T0, write_frame(tmp_path, T0))]
    engine, clock = FakeEngine([]), FakeClock()
    replayer = make_replayer(engine, clock, utcnow=lambda: T0)
    assert not replayer.run(frames, stop_at=T0 - timedelta(minutes=1))
    assert engine.predicted == []


def test_dry_run_sends_nothing(tmp_path):
    frames = [(T0, write_frame(tmp_path, T0))]
    engine, clock = FakeEngine([0.9]), FakeClock()
    make_replayer(engine, clock, send_alerts=False).run(frames)
    assert engine.sent == []


def test_replay_engine_never_sends_last_image(tmp_path):
    # A replayed frame is not current: it must never become the camera's last image
    engine = replay_folder.ReplayEngine(cache_folder=str(tmp_path), cam_creds={"CAM": ("token", 1)})
    assert engine._states["CAM"]["last_image_sent"] == float("inf")
    assert engine._new_state()["last_image_sent"] == float("inf")
