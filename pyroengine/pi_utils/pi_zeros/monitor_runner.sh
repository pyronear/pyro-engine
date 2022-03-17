#!/bin/bash
until python3 /home/pi/pi_zeros/runner.py; do
    echo "'runner.py' crashed with exit code $?. Restarting..." >&2
    sleep 1
done