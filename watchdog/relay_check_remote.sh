#!/usr/bin/env bash
# Run the full relay check from a remote machine.
# Mirrors the Ansible playbook logic without requiring Ansible.
#
# Usage:
#   ./relay_check_remote.sh [--trigger h|l]

set -euo pipefail

# ============================================================
# CONFIG — adjust to match your setup
# ============================================================

MAIN_PI_IP=""
PIZERO_IP="192.168.1.98"
PI_USER="pi"
TRIGGER="h"
MAIN_REPO_PATH="/home/pyro-engine"
PIZERO_REPO_PATH="/home/pi/pyro-engine"

MAIN_SCRIPT="$MAIN_REPO_PATH/watchdog/main_pi/relay_check.py"
PIZERO_SCRIPT="$PIZERO_REPO_PATH/watchdog/pi_zero/relay_check.py"

LOCAL_MAIN_LOG="/tmp/relay_check_main_pi.log"
PIZERO_LOG="/tmp/relay_check.log"

SSH_TIMEOUT=5
WAIT_SSH_RETRIES=30   # x WAIT_SSH_INTERVAL => max 150s to come back
WAIT_SSH_INTERVAL=5

# Pi Zero is on the local LAN — reachable only via the main Pi as jump host
SSH_MAIN="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=$SSH_TIMEOUT"
SSH_PIZERO="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=$SSH_TIMEOUT -J $PI_USER@$MAIN_PI_IP"

# ============================================================
# HELPERS
# ============================================================

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_ssh() {
    local ssh_cmd="$1" target="$2" label="$3" retries="$WAIT_SSH_RETRIES"
    log "Waiting for SSH on $label ($target) ..."
    for ((i = 1; i <= retries; i++)); do
        if $ssh_cmd -o BatchMode=yes "$PI_USER@$target" true 2>/dev/null; then
            log "SSH ready on $label ($target)"
            return 0
        fi
        log "  attempt $i/$retries — not ready yet, retrying in ${WAIT_SSH_INTERVAL}s ..."
        sleep "$WAIT_SSH_INTERVAL"
    done
    log "ERROR: $label ($target) did not become reachable in time"
    return 1
}

# ============================================================
# ARG PARSING
# ============================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --trigger) TRIGGER="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ "$TRIGGER" != "h" && "$TRIGGER" != "l" ]]; then
    echo "ERROR: --trigger must be h or l"
    exit 1
fi

# ============================================================
# STEP 1 — run relay_check on main Pi (blocking, ~2 min)
#          this power-cycles the Pi Zero and waits for it to return
# ============================================================

log "=== STEP 1: main Pi relay check (trigger=$TRIGGER) ==="
$SSH_MAIN "$PI_USER@$MAIN_PI_IP" "python3 $MAIN_SCRIPT --trigger $TRIGGER" | tee "$LOCAL_MAIN_LOG"
log "STEP 1 complete"

# ============================================================
# STEP 2 — wait for Pi Zero SSH via jump host (it just rebooted)
# ============================================================

log "=== STEP 2: waiting for Pi Zero SSH (via main Pi jump) ==="
wait_for_ssh "$SSH_PIZERO" "$PIZERO_IP" "Pi Zero"

# ============================================================
# STEP 3 — launch Pi Zero relay check detached (fire-and-forget)
#          this cuts power to the main Pi — SSH session dies intentionally
# ============================================================

log "=== STEP 3: launching Pi Zero relay check (--relay main) ==="
$SSH_PIZERO "$PI_USER@$PIZERO_IP" \
    "nohup python3 $PIZERO_SCRIPT --trigger $TRIGGER \
     > $PIZERO_LOG 2>&1 < /dev/null & disown"
log "Pi Zero test launched — main Pi will lose power and reboot"

# ============================================================
# STEP 4 — wait for main Pi to reboot and come back
# ============================================================

log "=== STEP 4: waiting for main Pi to reboot ==="
# brief pause to let power actually cut before we start polling
sleep 20
wait_for_ssh "$SSH_MAIN" "$MAIN_PI_IP" "main Pi"

# ============================================================
# STEP 5 — wait for Pi Zero to come back on the network
#          the cams relay also cuts the router (12V rail), so the
#          Pi Zero loses connectivity until power is restored
# ============================================================

log "=== STEP 5: waiting for Pi Zero network (router on 12V rail) ==="
wait_for_ssh "$SSH_PIZERO" "$PIZERO_IP" "Pi Zero"
# brief extra wait for the relay_check script to finish writing the log
sleep 15

# ============================================================
# STEP 6 — fetch and display logs
#          Pi Zero logs are fetched via main Pi jump (now back up)
# ============================================================

log "=== STEP 6: fetching logs ==="

echo ""
echo "────────────────────────────────────────────────────────"
echo "  LOG — main Pi (local): $LOCAL_MAIN_LOG"
echo "────────────────────────────────────────────────────────"
cat "$LOCAL_MAIN_LOG" || true

echo ""
echo "────────────────────────────────────────────────────────"
echo "  LOG — Pi Zero ($PIZERO_IP): $PIZERO_LOG"
echo "────────────────────────────────────────────────────────"
$SSH_PIZERO "$PI_USER@$PIZERO_IP" "cat $PIZERO_LOG" || true

echo ""
log "=== Done ==="
