# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

"""Check the image-coordinate widget against the upgraded Streamlit version."""

# ruff: file-ignore[pytest-unittest-assertion]

import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest


class TestImageWidget(unittest.TestCase):
    def test_clickable_snapshot(self):
        app_file = Path(__file__).resolve().parents[1] / "livestream_app.py"
        app = AppTest.from_file(str(app_file), default_timeout=20).run()
        with patch(
            "pyro_camera_api_client.PyroCameraAPIClient.capture_image",
            return_value=Image.new("RGB", (320, 240)),
        ):
            camera = next(widget for widget in app.text_input if widget.label == "Camera IP (manual)")
            camera.set_value("192.0.2.1").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.get("component_instance"))
