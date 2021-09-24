# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.
import io
import unittest
from threading import Thread
from unittest.mock import Mock, patch

# noinspection PyUnresolvedReferences
import pi_patch
import requests
from requests import RequestException

from pyroengine.pi_utils.pi_zeros.runner import Runner


class RunnerTester(unittest.TestCase):
    def setUp(self):
        self.module_path = "pyroengine.pi_utils.pi_zeros.runner"
        Runner.CAPTURE_DELAY = 0
        Runner.LOOP_INTERVAL = 0

    def test_capture_stream(self):
        with patch(f"{self.module_path}.picamera") as mock_picamera:
            runner = Runner("my_url")
            result = runner.capture_stream()

        mock_picamera.PiCamera.assert_called_once_with()
        mock_picamera.PiCamera.return_value.start_preview.assert_called_once_with()
        mock_picamera.PiCamera.return_value.capture.assert_called_once_with(
            result["file"], format="jpeg"
        )

    def test_send_stream_raises_request_exception_on_400(self):
        with patch(f"{self.module_path}.requests") as mock_requests:
            mock_response = Mock(spec=requests.Response)
            mock_requests.post.return_value = mock_response
            mock_response.raise_for_status.side_effect = RequestException()

            runner = Runner("my_url")

            files = {"file": io.BytesIO()}

            runner.send_stream(files)

            mock_response.raise_for_status.assert_called_once_with()

    def test_run(self):
        runner = Runner("my_url")
        calls = []

        def new_capture_stream():
            calls.append(True)
            if len(calls) == 3:
                runner.stop_runner()

        mock_capture_stream = Mock(side_effect=new_capture_stream)
        with patch.object(runner, "capture_stream", new=mock_capture_stream):
            runner.run()

        self.assertEqual(3, mock_capture_stream.call_count)

    def test_stop_runner(self):
        runner = Runner("my_url")
        monitoring_thread = Thread(target=lambda: runner.run(), daemon=True)
        with patch(f"{self.module_path}.requests") as mock_requests:
            mock_response = requests.Response()
            mock_response.status_code = 200
            mock_requests.post.return_value = mock_response

            monitoring_thread.start()

            runner.stop_runner()
            monitoring_thread.join(timeout=1)
            self.assertFalse(monitoring_thread.is_alive())


if __name__ == "__main__":
    unittest.main()
