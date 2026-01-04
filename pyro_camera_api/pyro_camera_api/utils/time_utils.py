# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


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
