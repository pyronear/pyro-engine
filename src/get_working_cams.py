import json
import logging
from typing import Optional

import cv2
from PIL import Image


# Define the RTSPCamera class
class RTSPCamera:
    """
    A class for interacting with cameras via RTSP.

    Attributes:
        rtsp_url (str): Full RTSP URL of the camera.
        ip_address (str): IP address of the camera.
        cam_type (str): Type of the camera.
    """

    def __init__(self, rtsp_url: str, ip_address: str, cam_type: str):
        self.rtsp_url = rtsp_url
        self.ip_address = ip_address
        self.cam_type = cam_type

    def capture(self) -> Optional[Image.Image]:
        """
        Captures an image from the camera and returns it as a PIL Image.

        Returns:
            Image.Image: The captured image, or None if an error occurred or timeout.
        """
        try:
            # Open the RTSP stream
            cap = cv2.VideoCapture(self.rtsp_url)

            if not cap.isOpened():
                logging.error("Unable to open RTSP stream.")
                return None

            # Read a single frame
            ret, frame = cap.read()

            if not ret:
                logging.error("Unable to read frame from RTSP stream.")
                return None

            # Convert the frame to a PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return image

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None


# Set up logging
logging.basicConfig(level=logging.INFO)

# Read the input JSON file
with open("data/all_credentials.json", "r") as infile:
    all_cameras = json.load(infile)

# Initialize a dictionary to store cameras with successful image capture
valid_cameras = {}

# Iterate through each camera entry
for camera_name, camera_info in all_cameras.items():
    rtsp_url = camera_info["rtsp_url"]
    credentials = camera_info["credentials"][0]  # Assume the first credentials entry is used
    ip_address = camera_info.get("ip_address", "N/A")
    cam_type = camera_info.get("type", "N/A")

    logging.info(f"Attempting to connect to camera: {camera_name}")

    # Create an RTSPCamera instance
    camera = RTSPCamera(rtsp_url, ip_address, cam_type)

    # Try to capture an image
    image = camera.capture()

    if image:
        logging.info(f"Successfully captured image from: {camera_name}")
        valid_cameras[camera_name] = camera_info
    else:
        logging.warning(f"Failed to capture image from: {camera_name}")

# Save the valid cameras to a new JSON file
with open("data/credentials.json", "w") as outfile:
    json.dump(valid_cameras, outfile, indent=4)

logging.info("Processing complete. Valid cameras saved to credentials.json.")
