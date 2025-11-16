# pyro_camera_api/utils/time_utils.py

from __future__ import annotations

import time

_last_command_time: float = time.time()


def update_command_time() -> None:
    """Record current timestamp when a control API call is triggered."""
    global _last_command_time
    _last_command_time = time.time()


def seconds_since_last_command() -> float:
    """Return how many seconds have passed since last command call."""
    return time.time() - _last_command_time
