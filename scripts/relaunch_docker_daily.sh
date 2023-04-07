#!/bin/bash
# This script must be run with a crontab, run every day at 1am to relaunch the docker
# 0 1 * * * bash /home/pi/pyro-engine/scripts/relaunch_docker_daily.sh


make stop -C /home/pi/pyro-engine/
make run -C /home/pi/pyro-engine/