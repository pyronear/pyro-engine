# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
from threading import Thread
from unittest.mock import Mock, patch

# noinspection PyUnresolvedReferences
import pi_patch
import requests
from requests import HTTPError

from pyroengine.pi_utils.pi_zeros.monitor_pi import MonitorPi


class MonitorPiTester(unittest.TestCase):
    def setUp(self):
        self.module_path = "pyroengine.pi_utils.pi_zeros.monitor_pi"

    def test_get_record(self):
        with patch(f"{self.module_path}.psutil") as mock_psutil, patch(
            f"{self.module_path}.requests"
        ) as mock_requests:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response

            # add "attributes" to psutil mock
            mock_psutil.cpu_percent.return_value = 11
            mock_psutil.virtual_memory().available = 99 * 1024**3

            record_logs = MonitorPi("my_url")
            record_logs.cpu_temp = Mock(temperature=5)

            record_logs.get_record()

        test_metrics = {
            "id": 0,
            "cpu_temperature_C": record_logs.cpu_temp.temperature,
            "mem_available_GB": mock_psutil.virtual_memory().available / 1024**3,
            "cpu_usage_percent": mock_psutil.cpu_percent(),
        }
        mock_requests.post.assert_called_once_with("my_url", json=test_metrics)

    def test_get_record_raises_http_exception_on_400(self):
        with patch(f"{self.module_path}.requests") as mock_requests:
            mock_response = requests.Response()
            mock_response.status_code = 400
            mock_requests.post.return_value = mock_response

            record_logs = MonitorPi("my_url")

            with self.assertRaises(HTTPError):
                record_logs.get_record()

    def test_record(self):
        record_logs = MonitorPi("my_url")
        calls = []

        def new_get_record():
            calls.append(True)
            if len(calls) == 3:
                record_logs.stop_monitoring()

        mock_get_record = Mock(side_effect=new_get_record)
        with patch.object(record_logs, "get_record", new=mock_get_record):
            record_logs.record(time_step=0)

        self.assertEqual(3, mock_get_record.call_count)

    def test_record_catches_exception(self):
        record_logs = MonitorPi("my_url")
        calls = []

        def new_get_record():
            calls.append(True)
            if len(calls) == 3:
                record_logs.stop_monitoring()
            elif len(calls) == 2:
                raise HTTPError()

        mock_get_record = Mock(side_effect=new_get_record)
        with patch.object(record_logs, "get_record", new=mock_get_record):
            record_logs.record(time_step=0)

        self.assertEqual(3, mock_get_record.call_count)

    def test_stop_monitoring(self):
        record_logs = MonitorPi("my_url")
        monitoring_thread = Thread(
            target=lambda: record_logs.record(time_step=0.1), daemon=True
        )
        with patch(f"{self.module_path}.requests") as mock_requests:
            mock_response = requests.Response()
            mock_response.status_code = 400
            mock_requests.post.return_value = mock_response

            monitoring_thread.start()

            record_logs.stop_monitoring()
            monitoring_thread.join(timeout=1)
            self.assertFalse(monitoring_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
