# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Replay the past frames of a folder camera through the engine, in capture order.

Frames are read from {folder_path}/{id}/{pose_id}/{YYYYMMDDTHHMMSSZ}.jpg, the layout of the
folder adapter, and analyzed the way the live loop would have: night frames skipped, the
detection history dropped after a few night frames in a row, detections uploaded with their
capture time as recorded_at.

The API groups detections by arrival time, not capture time. Replayed fast, detections hours
apart would join one sequence. After an upload, the replay therefore waits until
`--sequence-pause` seconds have passed before analyzing a frame captured more than
`--sequence-gap` seconds after it, so the sequence closes as it would have live.

Progress is saved after each frame and the replay resumes from it: it can run over several
nights (`--stop-at`), from a cron. It only stops between events, never in the middle of one.

Example, inside the engine container:
    python replay_folder_cam.py --until 2026-11-19T10:00:00Z --dry-run --limit 100 --interval 0
    python replay_folder_cam.py --until 2026-11-19T10:00:00Z --stop-at 05:00
"""

import argparse
import fcntl
import json
import logging
import os
import pathlib
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image

from pyroengine.core import NIGHT_FRAMES_BEFORE_RESET, is_day_time
from pyroengine.engine import Engine
from pyroengine.logs import setup_logging

logger = logging.getLogger("replay_folder_cam")

FRAME_NAME = re.compile(r"\d{8}T\d{6}Z\.jpg")
FRAME_TIME_FORMAT = "%Y%m%dT%H%M%SZ"
# Extra frames analyzed after --limit or --stop-at to let an ongoing event end
MAX_FRAMES_TO_FINISH_EVENT = 120


class ReplayEngine(Engine):
    """Engine that never uploads a replayed frame as the camera's last image (it is not current)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for state in self._states.values():
            state["last_image_sent"] = float("inf")

    def _new_state(self) -> Dict[str, Any]:
        state = super()._new_state()
        state["last_image_sent"] = float("inf")
        return state


def parse_time(value: str) -> datetime:
    """ISO 8601 time, UTC when no offset is given."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def list_frames(
    directory: pathlib.Path, since: Optional[datetime], until: datetime
) -> List[Tuple[datetime, pathlib.Path]]:
    """Frames captured after since (excluded) and before until (excluded), oldest first."""
    frames = []
    for path in directory.iterdir():
        if not FRAME_NAME.fullmatch(path.name):
            continue
        try:
            captured_at = datetime.strptime(path.name[:-4], FRAME_TIME_FORMAT).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if (since is None or captured_at > since) and captured_at < until:
            frames.append((captured_at, path))
    frames.sort()
    return frames


def load_progress(path: pathlib.Path) -> Optional[datetime]:
    if not path.is_file():
        return None
    return parse_time(json.loads(path.read_text())["last_frame"])


def save_progress(path: pathlib.Path, captured_at: datetime) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_frame": captured_at.isoformat()}))
    tmp.replace(path)


def next_occurrence(hhmm: str, now: datetime) -> datetime:
    """Next time it is hhmm (UTC) after now: 05:00 asked at 17:30 is tomorrow morning."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return target if target > now else target + timedelta(days=1)


class Replayer:
    def __init__(
        self,
        engine: Any,
        cam_id: str,
        interval: float,
        sequence_gap: float,
        sequence_pause: float,
        send_alerts: bool = True,
        progress_file: Optional[pathlib.Path] = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.engine = engine
        self.cam_id = cam_id
        self.interval = interval
        self.sequence_gap = timedelta(seconds=sequence_gap)
        self.sequence_pause = sequence_pause
        self.send_alerts = send_alerts
        self.progress_file = progress_file
        self.clock = clock
        self.sleep = sleep
        self.utcnow = utcnow
        # Capture time of the last uploaded frame, and when it was uploaded
        self.last_upload_capture: Optional[datetime] = None
        self.last_upload_at: float = 0.0
        self.night_streak = 0
        self.stats = {"analyzed": 0, "night": 0, "unreadable": 0, "detections": 0, "uploads": 0, "pauses": 0}

    def idle(self) -> bool:
        """No event ongoing, waiting for a temporal verdict or waiting for upload."""
        state = self.engine._states.get(self.cam_id) or {}
        return not state.get("ongoing") and not state.get("temporal_events") and not self.engine._alerts

    def _wait_for_sequence_end(self, captured_at: datetime) -> None:
        if self.last_upload_capture is None or captured_at - self.last_upload_capture <= self.sequence_gap:
            return
        remaining = self.sequence_pause - (self.clock() - self.last_upload_at)
        if remaining > 0:
            logger.info(
                "Next frame %s is %s after the last upload: waiting %.0fs for the API to close the sequence",
                captured_at.isoformat(),
                captured_at - self.last_upload_capture,
                remaining,
            )
            self.stats["pauses"] += 1
            self.sleep(remaining)
        self.last_upload_capture = None

    def _upload(self) -> None:
        if not self.engine._alerts:
            return
        last = max(parse_time(alert["ts"]) for alert in self.engine._alerts)
        before = len(self.engine._alerts)
        self.engine._process_alerts()
        self.stats["uploads"] += before - len(self.engine._alerts)
        self.last_upload_capture = max(last, self.last_upload_capture or last)
        self.last_upload_at = self.clock()

    def process(self, captured_at: datetime, path: pathlib.Path) -> None:
        """Analyze one frame, as the live loop would have."""
        self._wait_for_sequence_end(captured_at)
        start = self.clock()
        try:
            with Image.open(path) as source:
                frame = source.convert("RGB")
        except Exception as exc:
            logger.error("Could not read %s: %s", path, exc)
            self.stats["unreadable"] += 1
            return

        if not is_day_time(None, frame, "ir"):
            self.stats["night"] += 1
            self.night_streak += 1
            if self.night_streak == NIGHT_FRAMES_BEFORE_RESET:
                # As live: evening frames must not be combined with the next morning's ones
                self.engine._states.pop(self.cam_id, None)
            return

        self.night_streak = 0
        conf = self.engine.predict(frame, self.cam_id, captured_at=captured_at.isoformat())
        self.stats["analyzed"] += 1
        if conf > self.engine.conf_thresh:
            self.stats["detections"] += 1
        logger.info("%s confidence=%.2f", captured_at.isoformat(), conf)
        if self.send_alerts:
            self._upload()
        remaining = self.interval - (self.clock() - start)
        if remaining > 0:
            self.sleep(remaining)

    def run(
        self,
        frames: List[Tuple[datetime, pathlib.Path]],
        limit: Optional[int] = None,
        stop_at: Optional[datetime] = None,
    ) -> bool:
        """Replay frames in order. Returns True when every frame was replayed."""
        extra = 0
        for captured_at, path in frames:
            done_asked = (limit is not None and self.stats["analyzed"] >= limit) or (
                stop_at is not None and self.utcnow() >= stop_at
            )
            if done_asked:
                if self.idle():
                    logger.info("Stopping before %s", captured_at.isoformat())
                    return False
                if extra >= MAX_FRAMES_TO_FINISH_EVENT:
                    logger.warning("Stopping in the middle of an event before %s", captured_at.isoformat())
                    return False
                extra += 1
            self.process(captured_at, path)
            if self.progress_file is not None:
                save_progress(self.progress_file, captured_at)
        if self.send_alerts:
            self._upload()
        if not self.idle():
            logger.warning("Last frame replayed with an event still open: its last frames may not be sent")
        return True


def main(args: argparse.Namespace) -> int:
    load_dotenv(".env")
    setup_logging()

    with pathlib.Path(args.creds).open("rb") as json_file:
        camera_data = json.load(json_file)
    if args.camera is None:
        if len(camera_data) != 1:
            logger.error("Several cameras in %s, pick one with --camera: %s", args.creds, sorted(camera_data))
            return 1
        args.camera = next(iter(camera_data))
    cam = camera_data[args.camera]
    if cam.get("adapter") != "folder" or cam.get("type") == "ptz":
        logger.error("Camera %s is not a static folder camera", args.camera)
        return 1
    pose_id = cam["pose_ids"][0]
    folder = pathlib.Path(args.folder or pathlib.Path(cam["folder_path"]) / str(cam["id"]) / str(pose_id))

    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    progress_file = cache / "progress.json"
    # One replay at a time: a cron run must not overlap one still going
    lock = (cache / "replay.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another replay is running (%s)", cache / "replay.lock")
        return 1

    until = parse_time(args.until)
    since = parse_time(args.since) if args.since else None
    resumed = None if args.dry_run else load_progress(progress_file)
    if resumed is not None and (since is None or resumed > since):
        since = resumed
    frames = list_frames(folder, since, until)
    logger.info(
        "Replaying %d frames of %s from %s (after %s, before %s)%s",
        len(frames),
        args.camera,
        folder,
        since.isoformat() if since else "the first one",
        until.isoformat(),
        ", dry run: nothing sent" if args.dry_run else "",
    )
    if not frames:
        logger.info("Nothing left to replay")
        return 0

    api_url = None if args.dry_run else os.environ["API_URL"]
    temporal_api_url = None if args.dry_run else os.environ.get("TEMPORAL_API_URL") or None
    engine = ReplayEngine(
        model_path=args.model_path,
        conf_thresh=args.thresh,
        max_bbox_size=args.max_bbox_size,
        api_url=api_url,
        cam_creds=None if args.dry_run else {args.camera: (cam["token"], pose_id)},
        cache_folder=str(cache),
        nb_consecutive_frames=args.nb_consecutive_frames,
        frame_size=(720, 1280),
        cache_size=args.cache_size,
        jpeg_quality=80,
        day_time_strategy="ir",
        temporal_api_url=temporal_api_url,
    )
    replayer = Replayer(
        engine,
        args.camera,
        interval=args.interval,
        sequence_gap=args.sequence_gap,
        sequence_pause=args.sequence_pause,
        send_alerts=not args.dry_run,
        progress_file=None if args.dry_run else progress_file,
    )
    stop_at = next_occurrence(args.stop_at, datetime.now(timezone.utc)) if args.stop_at else None
    finished = replayer.run(frames, limit=args.limit, stop_at=stop_at)
    logger.info("%s: %s", "Replay finished" if finished else "Replay paused", replayer.stats)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replay the past frames of a folder camera through the engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--until", required=True, help="Replay frames captured before this time (ISO 8601, UTC)")
    parser.add_argument("--since", default=None, help="Replay frames captured after this time (ISO 8601, UTC)")
    parser.add_argument("--creds", default="data/credentials.json", help="Camera credentials")
    parser.add_argument("--camera", default=None, help="Camera key in the credentials, needed when there are several")
    parser.add_argument("--folder", default=None, help="Frame folder, instead of the one of the credentials")
    parser.add_argument("--cache", default="data/replay", help="Replay folder: progress, lock and alert cache")
    parser.add_argument("--interval", type=float, default=5.0, help="Minimum seconds between two analyzed frames")
    parser.add_argument(
        "--sequence-gap",
        type=float,
        default=120.0,
        help="Capture gap after an upload, in seconds, above which the next detection starts a new sequence",
    )
    parser.add_argument(
        "--sequence-pause",
        type=float,
        default=150.0,
        help="Seconds to leave after an upload before such a frame, for the API to close the sequence",
    )
    parser.add_argument("--stop-at", default=None, help="Stop at this time (HH:MM, UTC), between two events")
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many analyzed day frames")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only: nothing sent, progress not saved")
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument("--thresh", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max_bbox_size", type=float, default=0.4, help="Maximum bbox size")
    parser.add_argument("--nb-consecutive-frames", type=int, default=6, help="Consecutive frames combined")
    parser.add_argument("--cache-size", type=int, default=20, help="Maximum number of alerts kept in cache")
    sys.exit(main(parser.parse_args()))
