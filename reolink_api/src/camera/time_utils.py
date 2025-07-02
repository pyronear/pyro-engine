# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import time

_last_command_time = time.time()


def update_command_time():
    global _last_command_time
    _last_command_time = time.time()


def seconds_since_last_command() -> float:
    return time.time() - _last_command_time
