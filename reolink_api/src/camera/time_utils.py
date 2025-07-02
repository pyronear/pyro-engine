import time

_last_command_time = time.time()


def update_command_time():
    global _last_command_time
    _last_command_time = time.time()


def seconds_since_last_command() -> float:
    return time.time() - _last_command_time
