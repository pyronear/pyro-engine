# Copyright (C) 2020-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import logging
import threading
import time
from urllib.parse import quote

import av

from anonymizer.anonymizer_registry import set_result
from anonymizer.vision import Anonymizer
from camera.config import CAM_USER, CAM_PWD  # read credentials from config

ANONYMIZER_MODEL = Anonymizer()


def _open_rtsp_container(camera_ip: str):
    """
    Open a PyAV container on the RTSP substream for the given camera IP,
    using user and password from camera.config.
    """
    user_enc = quote(str(CAM_USER), safe="")
    pwd_enc = quote(str(CAM_PWD), safe="")
    rtsp_url = f"rtsp://{user_enc}:{pwd_enc}@{camera_ip}:554/h264Preview_01_sub"

    opts = {
        "rtsp_transport": "tcp",
        "fflags": "nobuffer",
        "reorder_queue_size": "0",
        "analyzeduration": "0",
        "probesize": "32",
    }
    container = av.open(rtsp_url, options=opts)
    stream = next(s for s in container.streams if s.type == "video")
    frames_iter = container.decode(video=stream.index)
    return container, frames_iter


def anonymizer_loop(camera_ip: str, stop_flag: threading.Event):
    logging.info(f"[{camera_ip}] Starting anonymizer loop")

    container = None
    frames_iter = None
    backoff = 0.5

    try:
        container, frames_iter = _open_rtsp_container(camera_ip)
        backoff = 0.5
    except Exception as e:
        logging.error(f"[{camera_ip}] RTSP open error: {e}")

    while not stop_flag.is_set():
        try:
            if frames_iter is None:
                time.sleep(backoff)
                try:
                    container, frames_iter = _open_rtsp_container(camera_ip)
                    backoff = 0.5
                except Exception as e:
                    logging.error(f"[{camera_ip}] RTSP reopen error: {e}")
                    backoff = min(backoff * 2.0, 5.0)
                    continue

            frame = next(frames_iter, None)
            if frame is None:
                # end of stream or decode issue, force reopen
                raise RuntimeError("RTSP decode returned no frame")

            img_bgr = frame.to_ndarray(format="bgr24")
            preds = ANONYMIZER_MODEL(img_bgr)  # list of dicts with cls, score, box
            set_result(camera_ip, preds)

        except Exception as e:
            logging.error(f"[{camera_ip}] Anonymizer loop error: {e}")
            # reset and reopen after a short pause
            try:
                if container is not None:
                    container.close()
            except Exception:
                pass
            container, frames_iter = None, None
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 5.0)

    # cleanup
    try:
        if container is not None:
            container.close()
    except Exception:
        pass

    logging.info(f"[{camera_ip}] Anonymizer loop stopped")
