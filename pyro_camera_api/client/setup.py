# Copyright (C) 2020-2025, Pyronear.

import os
from pathlib import Path

from setuptools import setup, find_packages

PKG_NAME = "pyro_camera_api_client"
VERSION = os.getenv("BUILD_VERSION", "0.1.0.dev0")

if __name__ == "__main__":
    print(f"Building wheel {PKG_NAME} {VERSION}")

    cwd = Path(__file__).parent.absolute()
    with cwd.joinpath("pyro_camera_api_client", "version.py").open("w", encoding="utf-8") as f:
        f.write(f"__version__ = '{VERSION}'\n")

    setup(
        name=PKG_NAME,
        version=VERSION,
        packages=["pyro_camera_api_client"],  # or packages=find_packages()
    )
