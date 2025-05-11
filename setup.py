# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


import os
from pathlib import Path
from setuptools import setup #, find_packages

PKG_NAME = "pyroengine"
VERSION = os.getenv("BUILD_VERSION", "3.0.1.dev0")


if __name__ == "__main__":
    print(f"Building wheel {PKG_NAME}-{VERSION}")

    cwd = Path(__file__).parent.absolute()
    pkg_dir = cwd.joinpath(PKG_NAME)
    
    with open(pkg_dir.joinpath("version.py"), "w", encoding="utf-8") as f:
        f.write(f"__version__ = '{VERSION}'\n")

    setup(
        name=PKG_NAME,
        version=VERSION,
        packages=[PKG_NAME],
    )
