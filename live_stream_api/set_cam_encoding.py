import requests
import json

# ----------------------- CONFIGURATION -----------------------

# Camera Credentials
CAMERA_IP = "192.168.1.12"  # Change to your camera's IP
USERNAME = "admin"  # Change to your username
PASSWORD = "@Pyronear"  # Change to your password

# Which stream to update: "mainStream" or "subStream"
STREAM_TO_UPDATE = "subStream"  # Options: "mainStream" or "subStream"

# New Encoding Settings
NEW_BITRATE = 512  # Main Stream: [1024,1536,2048,3072,4096,5120,6144,7168,8192]
# Sub Stream: [64,128,160,192,256,384,512]

NEW_FRAMERATE = 10  # Main Stream: [25,22,20,18,16,15,12,10,8,6,4,2]
# Sub Stream: [15,10,7,4]

NEW_GOP = 4  # Keyframe interval (1 to 4)

NEW_SIZE = "640*360"  # Main Stream options: "2304*1296", "2560*1440", "3840*2160"
# Sub Stream: "640*360"

# ---------------------------------------------------------------


# Function to get token
def get_token():
    url = f"https://{CAMERA_IP}/api.cgi?cmd=Login"
    payload = [
        {
            "cmd": "Login",
            "param": {
                "User": {"Version": "0", "userName": USERNAME, "password": PASSWORD}
            },
        }
    ]
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            url, json=payload, headers=headers, verify=False
        )  # Ignore SSL verification
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


# Function to get current encoding settings
def get_encoding_settings(token):
    url = f"https://{CAMERA_IP}/api.cgi?cmd=GetEnc&token={token}"
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


# Function to set encoding settings for selected stream
def set_stream_encoding(token):
    current_settings = get_encoding_settings(token)
    if not current_settings:
        return

    stream_settings = current_settings[STREAM_TO_UPDATE]

    # Keep existing values unless new ones are provided
    updated_stream = {
        "size": NEW_SIZE if NEW_SIZE else stream_settings["size"],
        "frameRate": NEW_FRAMERATE if NEW_FRAMERATE else stream_settings["frameRate"],
        "bitRate": NEW_BITRATE if NEW_BITRATE else stream_settings["bitRate"],
        "gop": NEW_GOP if NEW_GOP else stream_settings["gop"],
        "profile": stream_settings["profile"],
    }

    url = f"https://{CAMERA_IP}/api.cgi?cmd=SetEnc&token={token}"

    payload = [
        {
            "cmd": "SetEnc",
            "action": 0,
            "param": {
                "Enc": {"channel": 0, "audio": 0, STREAM_TO_UPDATE: updated_stream}
            },
        }
    ]

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, verify=False)
        data = response.json()
        if data[0]["code"] == 0:
            print(f"✅ {STREAM_TO_UPDATE} updated successfully: {updated_stream}")
        else:
            print("❌ Failed to update {STREAM_TO_UPDATE}:", data)
    except Exception as e:
        print("❌ Error:", e)


# ----------------------- MAIN EXECUTION -----------------------

if __name__ == "__main__":
    token = get_token()
    if token:
        set_stream_encoding(token)
