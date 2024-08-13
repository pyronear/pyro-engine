# Copyright (C) 2023-2024, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import argparse

from pyroengine.sensors import ReolinkCamera


def main():
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
            "delete_ptz_preset",
            "reboot_camera",
            "get_auto_focus",
            "set_auto_focus",
            "start_zoom_focus",
        ],
        help="Action to perform on the camera",
    )
    parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
    parser.add_argument("--username", required=True, help="Username for camera access")
    parser.add_argument("--password", required=True, help="Password for camera access")
    parser.add_argument("--type", required=True, choices=["static", "ptz"], help="Type of the camera")
    parser.add_argument("--protocol", help="Protocol (http or https)", default="http")
    parser.add_argument(
        "--pos_id", type=int, help="Position ID for moving the camera or capturing at a specific position", default=None
    )
    parser.add_argument("--operation", help="Operation type for moving the camera (e.g., 'Left', 'Right')")
    parser.add_argument("--speed", type=int, help="Speed of the PTZ movement", default=1)
    parser.add_argument("--duration", type=int, help="Duration in seconds for moving the camera", default=1)
    parser.add_argument("--disable_autofocus", action="store_true", help="Disable autofocus if set")
    parser.add_argument("--zoom_position", type=int, help="Zoom position for start_zoom_focus", default=None)

    args = parser.parse_args()
    print(args)

    # Create an instance of ReolinkCamera
    camera_controller = ReolinkCamera(
        ip_address=args.ip, username=args.username, password=args.password, cam_type=args.type, protocol=args.protocol
    )

    # Handling different actions
    if args.action == "capture":
        image = camera_controller.capture(pos_id=args.pos_id)
        if image is not None:
            image.save("im.jpg")
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
    elif args.action == "delete_ptz_preset":
        if args.pos_id is not None:
            camera_controller.delete_ptz_preset(idx=args.pos_id)
        else:
            print("Position ID must be provided for deleting a PTZ preset.")
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


if __name__ == "__main__":
    main()
