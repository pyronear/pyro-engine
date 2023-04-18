#!/bin/bash
# This script must be run with a crontab, run every day at 1am to relaunch the docker
# 0 1 * * * bash /home/pi/pyro-engine/scripts/relaunch_docker_daily.sh


docker restart pyro-engine_pyro-engine_1
