#!/bin/bash
# Ping main pi
#	- if success: ok exit
#	- if failure: reboot

set -u

PING_CMD=('ping' '-c' '1' '-W' '10' 'main-pi-hostname')

if "${PING_CMD[@]}";
then
    exit 0;
fi;

sudo reboot;