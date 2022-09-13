# Copyright (C) 2022, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


import argparse
import json
import logging
import os
import time
from pathlib import Path

import urllib3
from dotenv import load_dotenv

from pyroengine import Engine

from .controller import ReolinkCamera, SystemController

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(format="%(asctime)s | %(levelname)s: %(message)s", level=logging.INFO, force=True)


def main(args):
    print(args)

    # .env loading
    load_dotenv(".env")
    API_URL = os.environ.get("API_URL")
    LAT = float(os.environ.get("LAT"))
    LON = float(os.environ.get("LON"))
    assert isinstance(API_URL, str) and isinstance(LAT, float) and isinstance(LON, float)
    CAM_USER = os.environ.get("CAM_USER")
    CAM_PWD = os.environ.get("CAM_PWD")
    assert isinstance(CAM_USER, str) and isinstance(CAM_PWD, str)

    # Loading camera creds
    with open(args.creds, "rb") as json_file:
        cameras_credentials = json.load(json_file)

    # Check if model is available in cache
    cache = Path(args.cache)
    _model, _config = args.model, args.config
    if cache.is_dir():
        if cache.joinpath("model.onnx").is_file():
            _model = str(cache.joinpath("model.onnx"))
        if cache.joinpath("config.json").is_file():
            _config = str(cache.joinpath("config.json"))

    if isinstance(_model, str):
        logging.info(f"Loading model from: {_model}")

    engine = Engine(
        args.hub,
        args.thresh,
        API_URL,
        cameras_credentials,
        LAT,
        LON,
        frame_saving_period=args.save_period // args.period,
        model_path=_model,
        cfg_path=_config,
        cache_folder=args.cache,
    )

    sys_controller = SystemController(
        engine,
        [ReolinkCamera(_ip, CAM_USER, CAM_PWD) for _ip in cameras_credentials],
    )

    while True:
        start_ts = time.time()
        sys_controller.run()
        # Sleep only once all images are processed
        time.sleep(max(args.period - time.time() + start_ts, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Raspberry Pi system controller", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Model
    parser.add_argument("--hub", type=str, default="pyronear/rexnet1_3x", help="HF Hub repo to use")
    parser.add_argument("--model", type=str, default=None, help="Overrides the ONNX model")
    parser.add_argument("--config", type=str, default=None, help="Overrides the model config")
    parser.add_argument("--thresh", type=float, default=0.5, help="Confidence threshold")
    # Camera & cache
    parser.add_argument("--creds", type=str, default="data/credentials.json", help="Camera credentials")
    parser.add_argument("--cache", type=str, default="./data", help="Cache folder")
    # Time config
    parser.add_argument("--period", type=int, default=30, help="Number of seconds between each camera stream analysis")
    parser.add_argument("--save-period", type=int, default=3600, help="Number of seconds between each media save")
    args = parser.parse_args()

    main(args)
