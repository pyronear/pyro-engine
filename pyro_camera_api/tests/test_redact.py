# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import pytest

from pyro_camera_api.utils.redact import redact_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        # Plain credentials
        ("rtsp://admin:secret@10.0.0.1:554/h264", "rtsp://***:***@10.0.0.1:554/h264"),
        # A "@" inside the password must not end up in the output: only the last "@"
        # delimits the host, so splitting on the first one leaks the rest of the password.
        ("rtsp://admin:p@ssword@10.0.0.1/live", "rtsp://***:***@10.0.0.1/live"),
        ("rtsp://admin:@@@@10.0.0.1/live", "rtsp://***:***@10.0.0.1/live"),
        # User only, no password
        ("rtsp://admin@10.0.0.1/live", "rtsp://***:***@10.0.0.1/live"),
        # Query string is preserved
        ("http://user:pwd@cam/snap.cgi?channel=1", "http://***:***@cam/snap.cgi?channel=1"),
    ],
)
def test_redact_url_masks_credentials(url, expected):
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        # No credentials at all: must be returned untouched
        "rtsp://10.0.0.1:554/h264",
        "srt://1.2.3.4:8890?streamid=publish:cam",
        # "@" in the path is not userinfo, the URL must not be mangled
        "rtsp://10.0.0.1/live@2",
        "http://cam/snap@2x.jpg",
    ],
)
def test_redact_url_leaves_credential_free_urls_intact(url):
    assert redact_url(url) == url


@pytest.mark.parametrize("arg", ["-i", "-f", "mpegts", "-rtsp_transport", "tcp", "1500k"])
def test_redact_url_leaves_plain_arguments_intact(arg):
    """redact_url is mapped over whole ffmpeg command lines, so non-URL args must pass through."""
    assert redact_url(arg) == arg


def test_redact_url_never_echoes_password_fragments():
    """Guard against any future split-based regression leaking part of the password."""
    redacted = redact_url("rtsp://admin:sup3r@S3cret!@10.0.0.1/live")
    assert "sup3r" not in redacted
    assert "S3cret" not in redacted
    assert "10.0.0.1" in redacted
