#!/bin/bash
# This script fetches changes from the main branch and updates if there are any.
#
# This script must be run with a crontab, run every hour
# 0 * * * * bash /home/pi/pyro-engine/scripts/update_script.sh >> /home/pi/pyro-engine/logfile.log 2>&1

# Print current date and time
echo "$(date): Checking for updates"

# Navigate to the repository directory
cd /home/pi/pyro-engine

# Fetch main branch specifically and update local tracking
git fetch origin main:refs/remotes/origin/main

# Get the lamain commit hash of the current HEAD and the remote main branch
HEADHASH=$(git rev-parse HEAD)
UPSTREAMHASH=$(git rev-parse refs/remotes/origin/main)

# Compare hashes and update if they are different
if [ "$HEADHASH" != "$UPSTREAMHASH" ]
then
    echo "$(date): New changes detected! Updating and executing script..."    
    git pull origin main
    bash /home/pi/pyro-engine/scripts/debug_script.sh
    echo "$(date): Update done!"
else
    echo "$(date): No changes detected"
fi

