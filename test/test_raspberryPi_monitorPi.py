# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import pi_patch
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import os
import pandas as pd
from pyroengine.raspberryPi.monitorPi import MonitorPi


class MonitorPiTester(unittest.TestCase):

    @patch('pyronearEngine.raspberryPi.monitorPi.psutil')
    def setUp(self, mock_psutil):

        self.monitoringFolder = Path(__file__).parent / 'fixtures'
        self.logFile = "pi_perf_example.csv"
        self.path_file = os.path.join(self.monitoringFolder, self.logFile)

        def prepare_get_record(self):

            with patch.object(MonitorPi, "__init__", lambda x, y, z: None):
                with patch('pyronearEngine.raspberryPi.monitorPi.strftime') as mock_strftime:
                    mock_strftime.return_value = '2020-04-01 12:34:56'

                    # add "attributes" to psutil mock
                    mock_psutil.cpu_percent.return_value = 11
                    mock_psutil.virtual_memory().available = 99 * 1024 ** 3
                    # add cst attribute to strftime mock
                    recordlogs = MonitorPi(None, None)
                    # fake init of MonitorPi
                    recordlogs.cpu_temp = Mock(temperature=5)
                    recordlogs.logFile = self.logFile
                    recordlogs.monitoringFolder = self.monitoringFolder

                    recordlogs.get_record()

                    self.the_test_line = {"datetime": [mock_strftime()],
                                          "cpu_temperature_C": [recordlogs.cpu_temp.temperature],
                                          "mem_available_GB": [mock_psutil.virtual_memory().available / 1024 ** 3],
                                          "cpu_usage_percent": [mock_psutil.cpu_percent()]}

        self.get_test_record = prepare_get_record

        prepare_get_record(self)

    @patch('pyronearEngine.raspberryPi.monitorPi.psutil')
    def test_MonitorPi_create_logfile(self, mock_psutil):
        self.assertTrue(os.path.exists(self.path_file), "file does not exist")

    def test_MonitorPi_file_content(self):
        the_record = pd.read_csv(self.path_file)
        pd.testing.assert_frame_equal(pd.DataFrame(data=self.the_test_line), the_record)

    @patch('pyronearEngine.raspberryPi.monitorPi.psutil')
    def test_MonitorPi_update_logFile(self, mock_psutil):
        with patch.object(MonitorPi, "__init__", lambda x, y, z: None):
            mock_psutil.cpu_percent.return_value = 11
            mock_psutil.virtual_memory().available = 99 * 1024 ** 3
            self.get_test_record(self)

        the_record = pd.read_csv(self.path_file)

        df_test = pd.concat([pd.DataFrame(self.the_test_line),
                             pd.DataFrame(self.the_test_line)], ignore_index=True)

        pd.testing.assert_frame_equal(df_test, the_record)

    def tearDown(self):
        if os.path.exists(self.path_file):
            os.remove(self.path_file)


if __name__ == '__main__':
    unittest.main()
