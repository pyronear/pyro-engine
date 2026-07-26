# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import sys
from pathlib import Path

# Make the pyro_camera_api package importable when tests run from the
# repository root (the package is not installed in the dev environment).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
