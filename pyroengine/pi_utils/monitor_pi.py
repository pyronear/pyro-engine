# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

from gpiozero import CPUTemperature
from time import sleep, strftime
import psutil
import pandas as pd
import os


class MonitorPi:
    """This class aims to monitor some metrics from Raspberry Pi system.
    Example
    --------
    monitor = MonitorPi("/home/pi/")
    monitor.record(5) # record metrics every 5 seconds
    """

    def __init__(self, monitoring_folder, logs_csv="pi_perf.csv"):
        self.cpu_temp = CPUTemperature()
        self.monitoring_folder = monitoring_folder
        self.logs_csv = logs_csv

    def get_record(self):
        line = {
            "datetime": [strftime("%Y-%m-%d %H:%M:%S")],
            "cpu_temperature_C": [self.cpu_temp.temperature],
            "mem_available_GB": [psutil.virtual_memory().available / 1024 ** 3],
            "cpu_usage_percent": [psutil.cpu_percent()],
        }

        path_file = os.path.join(self.monitoring_folder, self.logs_csv)
        if os.path.isfile(path_file):
            pd.DataFrame(data=line).to_csv(path_file, header=False)
        else:
            pd.DataFrame(data=line).to_csv(path_file)

    def record(self, time_step):
        while True:
            self.get_record()
            sleep(time_step)


if __name__ == "__main__":
    # TODO: send file to the local API (use requests POST?)
    log_folder = "/home/pi/pi_logs/"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    monitor = MonitorPi(log_folder)
    monitor.record(30)
