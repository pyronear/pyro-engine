#!/bin/bash

# ==========================================================
# Script: check_internet_connection.sh
# Description:
#   This script checks for an active internet connection by 
#   pinging google.com. If successful, it checks if the VPN 
#   interface (tun0) is up and reachable. If the VPN is not 
#   reachable or no IP is found, it restarts the VPN service.
#   If the internet is not reachable, it tries restarting 
#   the network. If the issue persists after a delay, the 
#   system reboots.
#
# Cron setup:
#   To run this script every 10 minutes, add the following
#   line to your crontab (edit with `crontab -e`):
#
#   */10 * * * * bash /home/pi/pyro-engine/scripts/check_internet_connection.sh
# ==========================================================

# Fix PATH for cron
export PATH=/usr/sbin:/usr/bin:/sbin:/bin

# Log file
LOG_FILE="/home/pi/check_internet.log"

# Truncate log if it exceeds 10MB 
MAX_SIZE=$((10 * 1024 * 1024))  # 10 MB
if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -ge $MAX_SIZE ]; then
    echo "🚮 Log file exceeded 7MB, truncating..." > "$LOG_FILE"
fi

# Add timestamp to logs
echo "==== $(date) ====" >> "$LOG_FILE"

# Step 1: Ping Google
if ping -c 1 -W 10 google.com > /dev/null 2>&1; then
    echo "✅ Internet OK (google.com reachable)" >> "$LOG_FILE"

    # Get local IP on tun0
    iplocalhostvpn=$(ip -4 addr show dev tun0 | awk '/inet / {print $2}' | cut -d/ -f1)

    if [[ -n "$iplocalhostvpn" ]]; then
        echo "🔍 VPN IP detected: $iplocalhostvpn" >> "$LOG_FILE"

        if ping -c 1 -W 10 "$iplocalhostvpn" > /dev/null 2>&1; then
            echo "✅ VPN OK (IP reachable)" >> "$LOG_FILE"
            exit 0
        else
            echo "⚠️ VPN IP detected but unreachable, restarting VPN" >> "$LOG_FILE"
            sudo systemctl restart openvpn@client
            exit 0
        fi
    else
        echo "❌ No VPN IP found, restarting VPN" >> "$LOG_FILE"
        sudo systemctl restart openvpn@client
        exit 0
    fi
else
    echo "❌ Internet unreachable, trying to restart network" >> "$LOG_FILE"
    sudo systemctl restart dhcpcd || echo "⚠️ Failed to restart dhcpcd" >> "$LOG_FILE"
fi

# Wait before retrying
sleep 60

# Check again after waiting
if ping -c 1 -W 10 google.com > /dev/null 2>&1; then
    echo "✅ Internet is back after network restart" >> "$LOG_FILE"
    exit 0
else
    echo "🔥 Internet still down, rebooting the machine..." >> "$LOG_FILE"
    sudo reboot
fi
