#!/bin/bash
set -euo pipefail

SHELLY_IP="${SHELLY_IP:-192.168.1.97}"
SCRIPT_NAME="${SCRIPT_NAME:-pi_watchdog}"
SCRIPT_FILE="${SCRIPT_FILE:-watchdog.js}"

echo "Finding script named ${SCRIPT_NAME}"

SCRIPT_ID=$(curl -s "http://${SHELLY_IP}/rpc/Script.List" | python3 - "$SCRIPT_NAME" <<'PY'
import json
import sys

script_name = sys.argv[1]
data = json.load(sys.stdin)

for script in data.get("scripts", []):
    if script.get("name") == script_name:
        print(script["id"])
        sys.exit(0)

raise SystemExit(f"Script not found: {script_name}")
PY
)

echo "Updating script id ${SCRIPT_ID}"

SCRIPT_ID="$SCRIPT_ID" SCRIPT_FILE="$SCRIPT_FILE" python3 - <<'PY'
import json
import os

script_id = int(os.environ["SCRIPT_ID"])
script_file = os.environ["SCRIPT_FILE"]

with open(script_file, "r", encoding="utf-8") as f:
    code = f.read()

with open("payload.json", "w", encoding="utf-8") as f:
    json.dump({"id": script_id, "code": code}, f)
PY

echo "Stopping script"
curl -s "http://${SHELLY_IP}/rpc/Script.Stop?id=${SCRIPT_ID}" || true
echo

echo "Uploading code"
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d @payload.json \
  "http://${SHELLY_IP}/rpc/Script.PutCode"
echo

echo "Enabling script on startup"
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d "{\"id\":${SCRIPT_ID},\"config\":{\"enable\":true}}" \
  "http://${SHELLY_IP}/rpc/Script.SetConfig"
echo

echo "Starting script"
curl -s "http://${SHELLY_IP}/rpc/Script.Start?id=${SCRIPT_ID}"
echo

echo "Script status"
curl -s "http://${SHELLY_IP}/rpc/Script.GetStatus?id=${SCRIPT_ID}"
echo
