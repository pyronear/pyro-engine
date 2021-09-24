#!/bin/bash
# This script performs:
# Ping google.com:
#	- if success:
#		Ping internal vpn ip:
#		- if success: ok exit
#		- if failure: restart vpn
#
#	- if failure: restart of network interfaces
# Sleep 60 sec
# Ping google again (to make sure that we have internet):
#	- if success: ok exit
#	- if failure: reboot

set -u

PING_CMD=('ping' '-c' '1' '-W' '10' 'google.com')

if "${PING_CMD[@]}";
then
    iplocalhostvpn=$(ifconfig tun0 | awk '/inet / {print $2}')
    PING_CMD_VPN=('ping' '-c' '1' '-W' '10' "$iplocalhostvpn")
    if "${PING_CMD_VPN[@]}";
    then
          exit 0;
    fi;

    sudo systemctl restart openvpn@client;
fi;

sudo service networking restart;
sleep 60;

if "${PING_CMD[@]}";
then
    exit 0;
fi;

python3 /home/pi/pyro-engine/pyroengine/pi_utils/reboot_router;
sleep 60;

sudo reboot;
