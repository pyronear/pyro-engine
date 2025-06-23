#!/bin/bash

# ==========================================================
# Script: check_internet_connection.sh
# Description:
#   Checks internet & VPN connection, restarts VPN service
#   or reboots router/machine if needed.
#
# .env example (for Teltonika RUT200 router):
#   ENABLE_ROUTER_REBOOT=true
#   ROUTER_IP=192.168.1.1
#   ROUTER_USER=root
#
# Cron setup (runs every 10 min):
#   */10 * * * * bash /home/pi/pyro-engine/scripts/check_internet_connection.sh
# ==========================================================

# Fix PATH for cron
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

# Log file
LOG_FILE="/home/pi/check_internet.log"

# Load .env if it exists
ENV_FILE="/home/engine/.env"
if [ -f "$ENV_FILE" ]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
    echo "📄 Loaded config from $ENV_FILE" >> "$LOG_FILE"
else
    echo "⚠️ No .env file found at $ENV_FILE, using defaults" >> "$LOG_FILE"
fi

# Truncate log if it exceeds 10MB 
MAX_SIZE=$((10 * 1024 * 1024))
if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -ge $MAX_SIZE ]; then
    echo "🚮 Log file exceeded 10MB, truncating..." > "$LOG_FILE"
fi

echo "==== $(date) ====" >> "$LOG_FILE"
echo "ENV DEBUG: ENABLE_ROUTER_REBOOT=$ENABLE_ROUTER_REBOOT, ROUTER_USER=$ROUTER_USER" >> "$LOG_FILE"

# Default values if not set
ENABLE_ROUTER_REBOOT=$(echo "$ENABLE_ROUTER_REBOOT" | tr '[:upper:]' '[:lower:]')
ROUTER_IP=${ROUTER_IP:-192.168.1.1}
ROUTER_USER=${ROUTER_USER:-root}

# Step 1: Ping Google
if ping -c 1 -W 10 google.com > /dev/null 2>&1; then
    echo "✅ Internet OK (google.com reachable)" >> "$LOG_FILE"

    iplocalhostvpn=$(ip -4 addr show dev tun0 | awk '/inet / {print $2}' | cut -d/ -f1)

    if [[ -n "$iplocalhostvpn" ]]; then
        echo "🔍 VPN IP detected: $iplocalhostvpn" >> "$LOG_FILE"
        if ping -c 1 -W 10 "$iplocalhostvpn" > /dev/null 2>&1; then
            echo "✅ VPN OK (IP reachable)" >> "$LOG_FILE"
            exit 0
        else
            echo "⚠️ VPN IP unreachable, restarting VPN" >> "$LOG_FILE"
            sudo systemctl restart openvpn@client
            exit 0
        fi
    else
        echo "❌ No VPN IP found, restarting VPN" >> "$LOG_FILE"
        sudo systemctl restart openvpn@client
        exit 0
    fi

else
    echo "❌ Internet unreachable" >> "$LOG_FILE"

    if [ "$ENABLE_ROUTER_REBOOT" = true ]; then
        echo "🔁 Attempting to reboot router at $ROUTER_IP" >> "$LOG_FILE"
        if ping -c 1 -W 3 "$ROUTER_IP" > /dev/null 2>&1; then
            if sshpass -p "$ROUTER_PASSWORD" ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$ROUTER_USER@$ROUTER_IP" "reboot" >> "$LOG_FILE" 2>&1; then
                echo "✅ Router reboot command sent" >> "$LOG_FILE"
            else
                echo "❌ Failed to send router reboot command (wrong password or SSH error)" >> "$LOG_FILE"
            fi

        else
            echo "⚠️ Router at $ROUTER_IP not reachable, skipping reboot" >> "$LOG_FILE"
        fi
    else
        echo "🔒 Router reboot disabled (ENABLE_ROUTER_REBOOT=$ENABLE_ROUTER_REBOOT)" >> "$LOG_FILE"
    fi
fi



# Wait 3 minutes and check again
sleep 180

if ping -c 1 -W 10 google.com > /dev/null 2>&1; then
    echo "✅ Internet is back after waiting" >> "$LOG_FILE"
    exit 0
else
    echo "🔥 Internet still down, rebooting the machine..." >> "$LOG_FILE"
    sudo reboot
fi
