# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import os
from pathlib import Path

from setuptools import find_packages, setup

PKG_NAME = "pyroengine"
VERSION = os.getenv("BUILD_VERSION", "3.0.0")

print(f"Building wheel {PKG_NAME}-{VERSION}")

cwd = Path(__file__).parent.absolute()
pkg_dir = cwd.joinpath(PKG_NAME)

with open(pkg_dir.joinpath("version.py"), "w", encoding="utf-8") as f:
    f.write(f"__version__ = '{VERSION}'\n")

setup(
    name=PKG_NAME,
    version=VERSION,
    packages=find_packages(),
)
