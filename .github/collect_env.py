# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
Based on https://github.com/pytorch/pytorch/blob/master/torch/utils/collect_env.py
This script outputs relevant system environment info
Run it with `python collect_env.py`.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import locale
import re
import subprocess
import sys
from collections import namedtuple

try:
    import pyroengine

    ENGINE_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    ENGINE_AVAILABLE = False

try:
    import onnxruntime

    ONNX_AVAILABLE = True
except (ImportError, NameError, AttributeError):
    ONNX_AVAILABLE = False

PY3 = sys.version_info >= (3, 0)


# System Environment Information
SystemEnv = namedtuple(
    "SystemEnv",
    [
        "pyroengine_version",
        "onnxruntime_version",
        "os",
        "python_version",
    ],
)


def run(command):
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    rc = p.returncode
    if PY3:
        enc = locale.getpreferredencoding()
        output = output.decode(enc)
        err = err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(run_lambda, command):
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(run_lambda, command, regex):
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def get_platform():
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform.startswith("win32"):
        return "win32"
    if sys.platform.startswith("cygwin"):
        return "cygwin"
    if sys.platform.startswith("darwin"):
        return "darwin"
    return sys.platform


def get_mac_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda):
    return run_and_read_all(run_lambda, "wmic os get Caption | findstr /v Caption")


def get_lsb_version(run_lambda):
    return run_and_parse_first_match(run_lambda, "lsb_release -a", r"Description:\t(.*)")


def check_release_file(run_lambda):
    return run_and_parse_first_match(run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"')


def get_os(run_lambda):
    platform = get_platform()

    if platform == "win32" or platform == "cygwin":
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return "Mac OSX {}".format(version)

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return desc

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return desc

        return platform

    # Unknown platform
    return platform


def get_env_info():
    run_lambda = run

    if ENGINE_AVAILABLE:
        pyroengine_str = pyroengine.__version__
    else:
        pyroengine_str = "N/A"

    if ONNX_AVAILABLE:
        onnxruntime_str = onnxruntime.__version__
    else:
        onnxruntime_str = "N/A"

    return SystemEnv(
        pyroengine_version=pyroengine_str,
        onnxruntime_version=onnxruntime_str,
        python_version=".".join(map(str, sys.version_info[:3])),
        os=get_os(run_lambda),
    )


env_info_fmt = """
PyroEngine version: {pyroengine_version}
ONNX runtime version: {onnxruntime_version}

OS: {os}

Python version: {python_version}
""".strip()


def pretty_str(envinfo):
    def replace_nones(dct, replacement="Could not collect"):
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(dct, true="Yes", false="No"):
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    mutable_dict = envinfo._asdict()

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info():
    """Collects environment information for debugging purposes

    Returns:
        str: environment information
    """
    return pretty_str(get_env_info())


def main():
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)


if __name__ == "__main__":
    main()
