# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

from gpiozero import CPUTemperature
from time import sleep, strftime
import psutil
import pandas as pd
import os


class MonitorPi:
    """This class aims to monitor some metrics from Raspberry Pi system.

    Example
    --------
    recordlogs = MonitorPi("/home/pi/Desktop/")
    recordlogs.record(5) # record metrics every 5 seconds
    """

    def __init__(self, monitoringFolder, logFile="pi_perf.csv"):

        # cpu temperature
        self.cpu_temp = CPUTemperature()

        # path
        self.monitoringFolder = monitoringFolder
        self.logFile = logFile

    def get_record(self):

        line = {"datetime": [strftime("%Y-%m-%d %H:%M:%S")],
                "cpu_temperature_C": [self.cpu_temp.temperature],
                "mem_available_GB": [psutil.virtual_memory().available / 1024 ** 3],
                "cpu_usage_percent": [psutil.cpu_percent()]
                }

        path_file = os.path.join(self.monitoringFolder, self.logFile)

        if os.path.isfile(path_file):
            pd.DataFrame(data=line).to_csv(path_file, mode='a', header=False, index=0)
        else:
            pd.DataFrame(data=line).to_csv(path_file, mode='a', index=0)

    def record(self, timeStep):

        while True:

            self.get_record()

            sleep(timeStep)


if __name__ == "__main__":

    recordlogs = MonitorPi("/home/pi/Desktop/")
    recordlogs.record(30)
