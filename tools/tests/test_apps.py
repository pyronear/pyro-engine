# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Guard the Streamlit and image-coordinate integration without camera hardware."""

# unittest keeps these checks runnable without adding pytest to the tools environment.
# ruff: noqa: PT009

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image
from streamlit.testing.v1 import AppTest

TOOLS = Path(__file__).resolve().parents[1]


class TestApps(unittest.TestCase):
    def test_disconnected_apps(self):
        for name in ("ptz_calibration_app.py", "livestream_app.py"):
            with self.subTest(app=name):
                app = AppTest.from_file(str(TOOLS / name), default_timeout=20).run()
                self.assertFalse(app.exception)
                self.assertTrue(app.info)

    def test_connected_calibration_tabs(self):
        app = AppTest.from_file(str(TOOLS / "ptz_calibration_app.py"), default_timeout=20)
        client = MagicMock()
        client.get_patrol_status.return_value = {}
        client.get_presets.return_value = []
        app.session_state["client"] = client
        app.session_state["live_img"] = Image.new("RGB", (320, 240))
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.tabs), 6)
        client.move.assert_not_called()
        client.zoom.assert_not_called()

    def test_livestream_renders_clickable_snapshot(self):
        app = AppTest.from_file(str(TOOLS / "livestream_app.py"), default_timeout=20).run()
        with patch(
            "pyro_camera_api_client.PyroCameraAPIClient.capture_image",
            return_value=Image.new("RGB", (320, 240)),
        ):
            camera = next(widget for widget in app.text_input if widget.label == "Camera IP (manual)")
            camera.set_value("192.0.2.1").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get("component_instance"))


if __name__ == "__main__":
    unittest.main()
