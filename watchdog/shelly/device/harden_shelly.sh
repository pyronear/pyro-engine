#!/bin/bash
set -euo pipefail

# Harden a Shelly Pro that is used only to run the pi_watchdog script.
# Disables cloud / remote / radio features the local watchdog does not need.
#
# WARNING: this turns off the Wi-Fi access point. Afterwards the device is only
# reachable over your main Wi-Fi (station mode, static IP). If that link is lost
# you will need the physical reset button to bring the AP back.

SHELLY_IP="${SHELLY_IP:-192.168.1.97}"
BASE="http://${SHELLY_IP}/rpc"

set_config() {
  local method="$1" payload="$2"
  echo "-> ${method} ${payload}"
  curl -fsS -X POST -H "Content-Type: application/json" -d "${payload}" "${BASE}/${method}"
  echo
}

count_list() {
  # $1 = RPC method, $2 = JSON array key
  curl -fsS "${BASE}/$1" | python3 -c 'import sys, json; print(len(json.load(sys.stdin).get(sys.argv[1], [])))' "$2"
}

echo "Hardening Shelly at ${SHELLY_IP}"

echo "== Cloud off =="
set_config Cloud.SetConfig '{"config":{"enable":false}}'

echo "== MQTT off =="
set_config MQTT.SetConfig '{"config":{"enable":false}}'

echo "== Outbound WebSocket off =="
set_config Ws.SetConfig '{"config":{"enable":false}}'

echo "== Wi-Fi access point off =="
set_config WiFi.SetConfig '{"config":{"ap":{"enable":false}}}'

echo "== BLE off =="
set_config BLE.SetConfig '{"config":{"enable":false,"rpc":{"enable":false}}}'

echo "== Webhooks (delete if any) =="
hooks=$(count_list Webhook.List hooks)
if [ "$hooks" -gt 0 ]; then
  echo "Found ${hooks} webhook(s), deleting all"
  curl -fsS -X POST -H "Content-Type: application/json" -d '{}' "${BASE}/Webhook.DeleteAll"
  echo
else
  echo "No webhooks, nothing to do"
fi

echo "== Schedules (delete if any) =="
jobs=$(count_list Schedule.List jobs)
if [ "$jobs" -gt 0 ]; then
  echo "Found ${jobs} schedule(s), deleting all"
  curl -fsS -X POST -H "Content-Type: application/json" -d '{}' "${BASE}/Schedule.DeleteAll"
  echo
else
  echo "No schedules, nothing to do"
fi

echo "Done."
