#!/bin/bash
# This script fetches changes from the develop branch and updates if there are any.
#
# This script must be run with a crontab, run every hour
# 0 * * * * bash /home/pi/pyro-engine/scripts/update_script.sh >> /home/pi/pyro-engine/logfile.log 2>&1

# Print current date and time
echo "$(date): Checking for updates"

# Navigate to the repository directory
cd /home/pi/pyro-engine

# Fetch develop branch specifically and update local tracking
git fetch origin develop:refs/remotes/origin/develop

# Get the ladevelop commit hash of the current HEAD and the remote develop branch
HEADHASH=$(git rev-parse HEAD)
UPSTREAMHASH=$(git rev-parse refs/remotes/origin/develop)

# Compare hashes and update if they are different
if [ "$HEADHASH" != "$UPSTREAMHASH" ]
then
    echo "$(date): New changes detected! Updating and executing script..."    
    git pull origin develop
    bash /home/pi/pyro-engine/scripts/debug_script.sh
    echo "$(date): Update done!"
else
    echo "$(date): No changes detected"
fi

