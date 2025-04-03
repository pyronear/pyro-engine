import os
import time
import argparse
from dotenv import load_dotenv
from pyroengine.sensors import ReolinkCamera

# Chargement des identifiants depuis le .env
load_dotenv()
CAM_USER = os.getenv("CAM_USER", "admin")
CAM_PWD = os.getenv("CAM_PWD", "@Pyronear")
PROTOCOL = "http"

# Parser l'adresse IP depuis les arguments CLI
parser = argparse.ArgumentParser(description="Sweep through focus values and capture images.")
parser.add_argument("--ip", required=True, help="IP address of the Reolink camera")
args = parser.parse_args()
CAM_IP = args.ip

# Intervalle de focus à tester
focus_values = list(range(680, 720, 1))  # Tu peux ajuster

# Dossier de sortie
output_dir = "focus_tests"
os.makedirs(output_dir, exist_ok=True)

# Créer la caméra
cam = ReolinkCamera(
    ip_address=CAM_IP,
    username=CAM_USER,
    password=CAM_PWD,
    protocol=PROTOCOL,
)

# Désactiver l'autofocus
cam.set_auto_focus(disable=True)
time.sleep(1)

# Boucle de test
for focus in focus_values:
    print(f"🔧 Setting focus to {focus}")
    cam.set_manual_focus(position=focus)
    time.sleep(2)  # Laisse le temps à la caméra de faire la mise au point
    img = cam.capture()
    if img:
        path = os.path.join(output_dir, f"focus_{focus}.jpg")
        img.resize((1280, 720)).save(path)
        print(f"📸 Saved image at {path}")
    else:
        print(f"⚠️ Failed to capture image at focus {focus}")
