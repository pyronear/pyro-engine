# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


"""Cooperative cancellation of the autofocus search.

The autofocus search (adapter focus_finder) sweeps the focus motor for
minutes while holding the per-camera move lock. Live streaming has priority:
stream startup sets the per-camera cancel event so a running search aborts at
its next step and restores the pre-search focus, then waits for the move lock
to be released before starting the pipeline, so the viewer never sees a
focus sweep.
"""

from __future__ import annotations

import logging

from pyro_camera_api.camera.registry import FOCUS_CANCEL_EVENTS, MOVE_LOCKS
from pyro_camera_api.services.stream import is_camera_streaming

logger = logging.getLogger(__name__)

# Seconds stream startup waits for a running focus operation to abort
FOCUS_CANCEL_TIMEOUT = 20.0


def stream_is_active(camera_id: str) -> bool:
    """True if a live pipeline or ffmpeg restream is running for this camera."""
    return is_camera_streaming(camera_id)


def focus_abort_requested(camera_id: str) -> bool:
    """Combined abort predicate: stream start pending or stream active."""
    if FOCUS_CANCEL_EVENTS[camera_id].is_set():
        return True
    return stream_is_active(camera_id)


def cancel_focus_and_wait(camera_id: str, timeout: float = FOCUS_CANCEL_TIMEOUT) -> bool:
    """
    Ask any running focus operation on this camera to abort and wait for it.

    Sets the per-camera cancel event (polled between steps by the focus
    search) then waits for MOVE_LOCKS to be free, which also covers the final
    focus restoration. Returns True when the camera is free.
    """
    event = FOCUS_CANCEL_EVENTS[camera_id]
    event.set()
    lock = MOVE_LOCKS[camera_id]
    acquired = lock.acquire(timeout=timeout)
    if acquired:
        lock.release()
    event.clear()
    return acquired
