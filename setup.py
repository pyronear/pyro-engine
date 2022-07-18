#!usr/bin/python

# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
import subprocess
from setuptools import setup, find_packages
from pathlib import Path

version = "0.0.1"
sha = "Unknown"
src_folder = "pyroengine"
package_index = "pyroengine"

cwd = Path(__file__).parent.absolute()

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION")
else:
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
    except Exception:
        pass
    if sha != "Unknown":
        version += "+" + sha[:7]
print(f"Building wheel {package_index}-{version}")

with open(cwd.joinpath(src_folder, "version.py"), "w") as f:
    f.write(f"__version__ = '{version}'\n")

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    # Metadata
    name=package_index,
    version=version,
    author="PyroNear Contributors",
    author_email="pyronear.d4g@gmail.com",
    maintainer="Pyronear",
    description="Pyronear Engine is a repository that aims at deploying pyronear",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/pyronear/pyro-engine",
    download_url="https://github.com/pyronear/pyro-engine/tags",
    license="Apache 2.0",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "pytorch",
        "deep learning",
        "vision",
        "models",
        "wildfire",
        "object detection",
    ],
    # Package info
    packages=find_packages(exclude=("test",)),
    zip_safe=True,
    python_requires=">=3.6.0",
    include_package_data=True,
    install_requires=requirements,
    package_data={"": ["LICENSE"]},
)
