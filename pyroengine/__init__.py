from typing import Any

from pyro_predictor import Predictor
from .version import __version__


# Lazy imports: core (SystemController, is_day_time) and engine require
# requests/pyroclient which are optional if only Predictor is used.
def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name in ("SystemController", "is_day_time"):
        from .core import SystemController, is_day_time

        globals()["SystemController"] = SystemController
        globals()["is_day_time"] = is_day_time
        return globals()[name]
    if name in ("engine", "core", "utils", "predictor"):
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'pyroengine' has no attribute {name!r}")
