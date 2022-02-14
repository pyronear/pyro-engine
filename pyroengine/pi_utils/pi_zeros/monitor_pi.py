# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
import os
from time import sleep

import psutil

import requests
from dotenv import load_dotenv
from gpiozero import CPUTemperature
from requests import RequestException

load_dotenv()

WEBSERVER_IP = os.environ.get("WEBSERVER_IP")
WEBSERVER_PORT = os.environ.get("WEBSERVER_PORT")


class MonitorPi:
    """This class aims to monitor some metrics from Raspberry Pi system.
    Example
    --------
    monitor = MonitorPi(url_of_webserver)
    monitor.record(5) # record metrics every 5 seconds
    """

    logger = logging.getLogger(__name__)

    def __init__(self, webserver_url):
        """Initialize parameters for MonitorPi."""
        self.cpu_temp = CPUTemperature()
        self.webserver_url = webserver_url
        self.is_running = True

    def get_record(self):
        metrics = {
            "id": 0,
            "cpu_temperature_C": self.cpu_temp.temperature,
            "mem_available_GB": psutil.virtual_memory().available / 1024 ** 3,
            "cpu_usage_percent": psutil.cpu_percent(),
        }

        response = requests.post(self.webserver_url, json=metrics)
        response.raise_for_status()

    def record(self, time_step):
        while self.is_running:
            try:
                self.get_record()
                sleep(time_step)
            except RequestException as e:
                self.logger.error(f"Unexpected error in get_record(): {e!r}")

    def stop_monitoring(self):
        self.is_running = False


if __name__ == "__main__":
    webserver_local_url = f"http://{WEBSERVER_IP}:{WEBSERVER_PORT}/metrics"
    monitor = MonitorPi(webserver_local_url)
    monitor.record(30)
