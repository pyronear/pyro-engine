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
#
# This script must be run with a crontab, run every 10 mn
# */10 * * * * bash /home/pi/pyro-engine/scripts/check_internet_connection.sh

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


sudo reboot;
