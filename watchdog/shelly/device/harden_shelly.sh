#!/bin/bash
set -euo pipefail

# Harden a Shelly Pro used only to run the pi_watchdog script.
# Disables cloud, remote access, radio features, webhooks and schedules that
# the local watchdog does not need, enables eco mode to reduce heat, and forces
# the watchdog outputs to power on at boot.
#
# The firmware may be a beta, so some RPC methods can be missing. Every call
# tolerates failure and the script keeps going.
#
# WARNING: this turns off the WiFi access point.
# Afterwards the device is only reachable over the main network.
# If that link is lost, physical access or a reset may be required.
#
# WARNING: this deletes ALL webhooks and schedules on the device. Only run it
# on a Shelly dedicated to the watchdog, not one shared with other automations.

SHELLY_IP="${SHELLY_IP:-192.168.1.97}"
BASE="http://${SHELLY_IP}/rpc"
OUT_DIR="${OUT_DIR:-.}"

# Outputs the watchdog manages; forced ON at boot so a Shelly restart does not
# leave the Pi / cameras powered off. Space-separated RPC ids.
WATCHDOG_OUTPUTS="${WATCHDOG_OUTPUTS:-0 1}"

CURL=(curl --connect-timeout 5 --max-time 20 -fsS)

rpc_post() {
  local method="$1" payload="$2" rc=0

  echo "-> ${method} ${payload}"
  "${CURL[@]}" -X POST -H "Content-Type: application/json" -d "${payload}" "${BASE}/${method}" || rc=$?
  echo
  if [ "$rc" -ne 0 ]; then
    echo "Warning: ${method} failed or is not supported on this firmware (curl exit ${rc})"
  fi
}

rpc_get() {
  "${CURL[@]}" "${BASE}/$1"
}

# Read-only call that never aborts the script. Optional second arg = output file.
rpc_get_optional() {
  local method="$1" outfile="${2:-}" rc=0

  if [ -n "$outfile" ]; then
    rpc_get "$method" > "$outfile" || rc=$?
  else
    rpc_get "$method" || rc=$?
  fi
  if [ "$rc" -ne 0 ]; then
    echo "Warning: ${method} failed or is not supported on this firmware (curl exit ${rc})"
  fi
}

count_list() {
  local method="$1" key="$2"

  rpc_get "$method" 2>/dev/null | python3 -c '
import json
import sys

key = sys.argv[1]
data = json.load(sys.stdin)
print(len(data.get(key, [])))
' "$key"
}

# Echo the number of items in an RPC list, or 0 if it cannot be determined.
safe_count() {
  local n
  n=$(count_list "$1" "$2" 2>/dev/null || true)
  [[ "$n" =~ ^[0-9]+$ ]] || n=0
  echo "$n"
}

echo "Hardening Shelly at ${SHELLY_IP}"

echo "== Device info =="
rpc_get_optional "Shelly.GetDeviceInfo"
echo

echo "== Available methods =="
rpc_get_optional "Shelly.ListMethods" "${OUT_DIR}/shelly_methods.json"
echo "Saved methods to ${OUT_DIR}/shelly_methods.json"

echo "== Cloud off =="
rpc_post "Cloud.SetConfig" '{"config":{"enable":false}}'

echo "== MQTT off =="
rpc_post "Mqtt.SetConfig" '{"config":{"enable":false}}'

echo "== Outbound WebSocket off =="
rpc_post "WS.SetConfig" '{"config":{"enable":false}}'

echo "== WiFi access point off =="
rpc_post "Wifi.SetConfig" '{"config":{"ap":{"enable":false}}}'

echo "== BLE off =="
rpc_post "BLE.SetConfig" '{"config":{"enable":false,"rpc":{"enable":false}}}'

echo "== Eco mode on =="
# Lowers CPU/radio activity when idle to reduce heat and power draw. The
# watchdog does not need fast RPC responses, so the extra latency is harmless.
rpc_post "Sys.SetConfig" '{"config":{"device":{"eco_mode":true}}}'

echo "== Outputs ON by default at boot =="
# Without this the outputs default to match_input and stay off after a Shelly
# restart, leaving the Pi / cameras unpowered until manual intervention.
for id in ${WATCHDOG_OUTPUTS}; do
  rpc_post "Switch.SetConfig" "{\"id\":${id},\"config\":{\"initial_state\":\"on\"}}"
done

echo "== Webhooks delete if any =="
hooks=$(safe_count "Webhook.List" "hooks")
if [ "$hooks" -gt 0 ]; then
  echo "Found ${hooks} webhook(s), deleting all"
  rpc_post "Webhook.DeleteAll" '{}'
else
  echo "No webhooks, nothing to do"
fi

echo "== Schedules delete if any =="
jobs=$(safe_count "Schedule.List" "jobs")
if [ "$jobs" -gt 0 ]; then
  echo "Found ${jobs} schedule(s), deleting all"
  rpc_post "Schedule.DeleteAll" '{}'
else
  echo "No schedules, nothing to do"
fi

echo "== Final config =="
rpc_get_optional "Shelly.GetConfig" "${OUT_DIR}/shelly_config_after_hardening.json"
echo "Saved final config to ${OUT_DIR}/shelly_config_after_hardening.json"

echo "Done."
