import json
import requests
import argparse
import os
from dotenv import load_dotenv

# ----------------------- CONFIGURATION -----------------------

# Load environment variables from .env file
load_dotenv()

CAM_USER = os.getenv("CAM_USER")
CAM_PWD = os.getenv("CAM_PWD")


# ----------------------- NEW ENCODING SETTINGS -----------------------

# Main Stream possible values:
# size: "2304*1296", "2560*1440", "3840*2160"
# bitRate (kbps): 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192
# frameRate (fps): 25, 22, 20, 18, 16, 15, 12, 10, 8, 6, 4, 2
# gop: 1, 2, 3, 4
NEW_SIZE_MAIN = "3840*2160"  # 4K resolution
NEW_BITRATE_MAIN = 4096  # kbps
NEW_FRAMERATE_MAIN = 15  # fps
NEW_GOP_MAIN = 2  # keyframe interval

# Sub Stream possible values:
# size: "640*360"
# bitRate (kbps): 64, 128, 160, 192, 256, 384, 512
# frameRate (fps): 15, 10, 7, 4
# gop: 1, 2, 3, 4
NEW_SIZE_SUB = "640*360"
NEW_BITRATE_SUB = 512
NEW_FRAMERATE_SUB = 10
NEW_GOP_SUB = 4

# ----------------------------------------------------------------------


def get_token(camera_ip):
    url = f"https://{camera_ip}/api.cgi?cmd=Login"
    payload = [
        {
            "cmd": "Login",
            "param": {"User": {"Version": "0", "userName": CAM_USER, "password": CAM_USER}},
        }
    ]
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, verify=False)
        data = response.json()
        if data[0]["code"] == 0:
            token = data[0]["value"]["Token"]["name"]
            print(f"✅ Token acquired: {token}")
            return token
        else:
            print("❌ Failed to get token:", data)
            return None
    except Exception as e:
        print("❌ Error:", e)
        return None


def get_encoding_settings(camera_ip, token):
    url = f"https://{camera_ip}/api.cgi?cmd=GetEnc&token={token}"
    try:
        response = requests.get(url, verify=False)
        data = response.json()
        if data[0]["code"] == 0:
            enc_settings = data[0]["value"]["Enc"]
            print(f"🔍 Current Encoding Settings: {enc_settings}")
            return enc_settings
        else:
            print("❌ Failed to get encoding settings:", data)
            return None
    except Exception as e:
        print("❌ Error:", e)
        return None


def set_both_streams_encoding(camera_ip, token):
    current_settings = get_encoding_settings(camera_ip, token)
    if not current_settings:
        return

    main_stream = current_settings["mainStream"]
    sub_stream = current_settings["subStream"]

    updated_main = {
        "size": NEW_SIZE_MAIN if NEW_SIZE_MAIN else main_stream["size"],
        "frameRate": NEW_FRAMERATE_MAIN if NEW_FRAMERATE_MAIN else main_stream["frameRate"],
        "bitRate": NEW_BITRATE_MAIN if NEW_BITRATE_MAIN else main_stream["bitRate"],
        "gop": NEW_GOP_MAIN if NEW_GOP_MAIN else main_stream["gop"],
        "profile": main_stream["profile"],
    }

    updated_sub = {
        "size": NEW_SIZE_SUB if NEW_SIZE_SUB else sub_stream["size"],
        "frameRate": NEW_FRAMERATE_SUB if NEW_FRAMERATE_SUB else sub_stream["frameRate"],
        "bitRate": NEW_BITRATE_SUB if NEW_BITRATE_SUB else sub_stream["bitRate"],
        "gop": NEW_GOP_SUB if NEW_GOP_SUB else sub_stream["gop"],
        "profile": sub_stream["profile"],
    }

    url = f"https://{camera_ip}/api.cgi?cmd=SetEnc&token={token}"

    payload = [
        {
            "cmd": "SetEnc",
            "action": 0,
            "param": {
                "Enc": {
                    "channel": 0,
                    "audio": 0,
                    "mainStream": updated_main,
                    "subStream": updated_sub,
                }
            },
        }
    ]

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, verify=False)
        data = response.json()
        if data[0]["code"] == 0:
            print(f"✅ Both streams updated successfully:\nMain Stream: {updated_main}\nSub Stream: {updated_sub}")
        else:
            print("❌ Failed to update streams:", data)
    except Exception as e:
        print("❌ Error:", e)


# ----------------------- MAIN EXECUTION -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update Reolink camera streams settings.")
    parser.add_argument("--ip", required=True, help="IP address of the camera")
    args = parser.parse_args()

    camera_ip = args.ip

    if not CAM_USER or not CAM_USER:
        print("❌ USERNAME or CAM_USER not found. Please set them in a .env file.")
        exit(1)

    token = get_token(camera_ip)
    if token:
        set_both_streams_encoding(camera_ip, token)
