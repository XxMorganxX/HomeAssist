#!/bin/bash
# ==============================================
# HomeAssist Background Service Runner
# ==============================================
# This script is called by launchd to run the assistant
# It loads environment variables, prevents system sleep via
# caffeinate, and starts the main process with proper cleanup.

set -e

# Project paths
PROJECT_DIR="/Users/morgannstuart/Desktop/HomeAssistV3"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
ENV_FILE="${PROJECT_DIR}/.env"
LOG_DIR="${PROJECT_DIR}/logs"

# PID tracking for cleanup
CAFFEINE_PID=""
ASSISTANT_PID=""

# ==============================================
# Cleanup function for graceful shutdown
# ==============================================
cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Received shutdown signal, cleaning up..."
    
    # Kill caffeinate if running
    if [ -n "$CAFFEINE_PID" ] && kill -0 "$CAFFEINE_PID" 2>/dev/null; then
        kill "$CAFFEINE_PID" 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopped caffeinate (PID: $CAFFEINE_PID)"
    fi
    
    # Kill assistant if running
    if [ -n "$ASSISTANT_PID" ] && kill -0 "$ASSISTANT_PID" 2>/dev/null; then
        kill "$ASSISTANT_PID" 2>/dev/null || true
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Stopped assistant (PID: $ASSISTANT_PID)"
    fi
    
    exit 0
}

# Register signal handlers
trap cleanup SIGTERM SIGINT SIGHUP

# ==============================================
# Setup
# ==============================================

# Change to project directory
cd "$PROJECT_DIR"

# Ensure logs directory exists
mkdir -p "$LOG_DIR"

# Load environment variables from .env file
if [ -f "$ENV_FILE" ]; then
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Loaded environment from .env"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# Verify venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Python venv not found at $VENV_PYTHON"
    exit 1
fi

# ==============================================
# Start caffeinate to prevent system sleep
# ==============================================
# Options:
#   -i : Prevent idle sleep (system won't sleep from inactivity)
#   -s : Prevent system sleep (only fully effective on AC power)
#   -d : Prevent display sleep (keeps display subsystem alive)
#   -m : Prevent disk from spinning down
#
# Note: On battery with lid closed, macOS may still enforce sleep
# due to thermal management. Use configure_power.sh for additional
# protection, or use a headless display adapter for guaranteed operation.

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting caffeinate to prevent sleep..."
caffeinate -i -s -d -m &
CAFFEINE_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Caffeinate started (PID: $CAFFEINE_PID)"

# Verify caffeinate is running
sleep 0.5
if ! kill -0 "$CAFFEINE_PID" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Caffeinate failed to start, continuing without sleep prevention"
    CAFFEINE_PID=""
fi

# ==============================================
# Run the assistant
# ==============================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting HomeAssist..."

# Run assistant in background so we can track its PID
"$VENV_PYTHON" -m assistant_framework &
ASSISTANT_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Assistant started (PID: $ASSISTANT_PID)"

# Wait for assistant to exit
wait $ASSISTANT_PID
EXIT_CODE=$?

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Assistant exited with code: $EXIT_CODE"

# Cleanup caffeinate
if [ -n "$CAFFEINE_PID" ] && kill -0 "$CAFFEINE_PID" 2>/dev/null; then
    kill "$CAFFEINE_PID" 2>/dev/null || true
fi

exit $EXIT_CODE
