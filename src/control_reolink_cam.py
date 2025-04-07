# Copyright (C) 2022-2025, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse
import os

from dotenv import load_dotenv

from pyroengine.sensors import ReolinkCamera


def main():
    """
    Control Reolink Camera for various operations.

    This script allows you to interact with a Reolink camera to perform various actions like capturing images,
    moving the camera, handling PTZ presets, and more.

    Available actions:
    - `capture`: Captures an image from the camera.
    - `move_camera`: Moves the camera in a specified direction or to a preset position.
    - `move_in_seconds`: Moves the camera in a specified direction for a certain duration.
    - `get_ptz_preset`: Retrieves the list of PTZ preset positions.
    - `set_ptz_preset`: Sets a PTZ preset position.
    - `reboot_camera`: Reboots the camera.
    - `get_auto_focus`: Retrieves the current auto-focus settings.
    - `set_auto_focus`: Enables or disables the auto-focus.
    - `start_zoom_focus`: Starts zooming the camera to a specific focus position.

    Examples:
        - Capture an image:
            python src/control_reolink_cam.py capture --ip 169.254.40.1

        - Move the camera to a preset position:
            python src/control_reolink_cam.py move_camera --ip 169.254.40.1 --operation ToPos --pos_id 10

        - Move the camera to the right for 3 seconds:
            python src/control_reolink_cam.py move_in_seconds --ip 169.254.40.1 --operation Right --duration 3

        - Get the list of PTZ presets:
            python src/control_reolink_cam.py get_ptz_preset --ip 169.254.40.1

        - Set a PTZ preset at position 1:
            python src/control_reolink_cam.py set_ptz_preset --ip 169.254.40.1 --pos_id 1

        - Reboot the camera:
            python src/control_reolink_cam.py reboot_camera --ip 169.254.40.1

        - Get the auto-focus settings:
            python src/control_reolink_cam.py get_auto_focus --ip 169.254.40.1

        - Disable auto-focus:
            python src/control_reolink_cam.py set_auto_focus --ip 169.254.40.1 --disable_autofocus

        - Enable auto-focus:
            python src/control_reolink_cam.py set_auto_focus --ip 169.254.40.1

        - Start zooming to zoom position 5:
            python src/control_reolink_cam.py start_zoom_focus --ip 169.254.40.1 --zoom_position 5

        - Set manual focus to position 20:
            python src/control_reolink_cam.py set_manual_focus --ip 169.254.40.1 --zoom_position 20

        - Manually focus and capture an image:
            python src/control_reolink_cam.py manual_focus_and_capture --ip 169.254.40.1 --zoom_position 20

        - Get current manual focus and zoom levels:
            python src/control_reolink_cam.py get_focus_level --ip 169.254.40.1
    """
    # Load environment variables
    load_dotenv()
    cam_user = os.getenv("CAM_USER")
    cam_pwd = os.getenv("CAM_PWD")

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Control Reolink Camera for various operations.")
    parser.add_argument(
        "action",
        choices=[
            "capture",
            "move_camera",
            "move_in_seconds",
            "get_ptz_preset",
            "set_ptz_preset",
            "reboot_camera",
            "get_auto_focus",
            "set_auto_focus",
            "start_zoom_focus",
            "manual_focus_and_capture",
            "set_manual_focus",
            "get_focus_level",
        ],
        help="Action to perform on the camera",
    )
    parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
    parser.add_argument("--username", help="Username for camera access", default=cam_user)
    parser.add_argument("--password", help="Password for camera access", default=cam_pwd)
    parser.add_argument("--protocol", help="Protocol (http or https)", default="http")
    parser.add_argument(
        "--pos_id", type=int, help="Position ID for moving the camera or capturing at a specific position", default=None
    )
    parser.add_argument("--operation", help="Operation type for moving the camera (e.g., 'Left', 'Right')")
    parser.add_argument("--speed", type=int, help="Speed of the PTZ movement", default=1)
    parser.add_argument("--duration", type=int, help="Duration in seconds for moving the camera", default=1)
    parser.add_argument("--disable_autofocus", action="store_true", help="Disable autofocus if set")
    parser.add_argument(
        "--zoom_position", type=int, help="Zoom position for start_zoom_focus or manual focus", default=None
    )

    args = parser.parse_args()
    print(args)

    # Create an instance of ReolinkCamera
    camera_controller = ReolinkCamera(
        ip_address=args.ip, username=args.username, password=args.password, protocol=args.protocol
    )

    # Handling different actions
    if args.action == "capture":
        image = camera_controller.capture(pos_id=args.pos_id)
        if image is not None:
            image.resize((1280, 720)).save("im.jpg")
            print("Image captured and saved as im.jpg")
        else:
            print("Failed to capture image.")
    elif args.action == "move_camera":
        if args.operation:
            camera_controller.move_camera(operation=args.operation, speed=args.speed, idx=args.pos_id)
        else:
            print("Operation type must be specified for moving the camera.")
    elif args.action == "move_in_seconds":
        if args.operation and args.duration:
            camera_controller.move_in_seconds(s=args.duration, operation=args.operation, speed=args.speed)
        else:
            print("Operation type and duration must be specified for moving the camera.")
    elif args.action == "get_ptz_preset":
        presets = camera_controller.get_ptz_preset()
        print("PTZ Presets:", presets)
    elif args.action == "set_ptz_preset":
        if args.pos_id is not None:
            camera_controller.set_ptz_preset(idx=args.pos_id)
        else:
            print("Position ID must be provided for setting a PTZ preset.")
    elif args.action == "reboot_camera":
        camera_controller.reboot_camera()
        print("Camera reboot initiated.")
    elif args.action == "get_auto_focus":
        autofocus_settings = camera_controller.get_auto_focus()
        print("AutoFocus Settings:", autofocus_settings)
    elif args.action == "set_auto_focus":
        camera_controller.set_auto_focus(disable=args.disable_autofocus)
        print(f"AutoFocus {'disabled' if args.disable_autofocus else 'enabled'}.")
    elif args.action == "start_zoom_focus":
        if args.zoom_position is not None:
            camera_controller.start_zoom_focus(position=args.zoom_position)
        else:
            print("Zoom position must be provided for starting zoom focus.")
    elif args.action == "manual_focus_and_capture":
        if args.zoom_position is None:
            print("Zoom position must be provided for manual focus capture.")
        else:
            camera_controller.set_auto_focus(disable=True)
            camera_controller.start_zoom_focus(position=args.zoom_position)
            print(f"Manual focus set at position {args.zoom_position}")
            image = camera_controller.capture(pos_id=args.pos_id)
            if image is not None:
                image.resize((1280, 720)).save("manual_focus.jpg")
                print("Captured image with manual focus and saved as manual_focus.jpg.")
            else:
                print("Failed to capture image.")

    elif args.action == "set_manual_focus":
        if args.zoom_position is None:
            print("You must provide --zoom_position for manual focus.")
        else:
            camera_controller.set_auto_focus(disable=True)
            camera_controller.set_manual_focus(position=args.zoom_position)
            print(f"Manual focus set at position {args.zoom_position}.")

    elif args.action == "get_focus_level":
        level = camera_controller.get_focus_level()
        if level:
            print(f"Current Focus Level: {level['focus']}, Zoom Level: {level['zoom']}")
        else:
            print("Failed to get focus level.")


if __name__ == "__main__":
    main()
