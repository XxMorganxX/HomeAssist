#!/bin/bash
# ==============================================
# HomeAssistV2 Background Service Runner
# ==============================================
# This script is called by launchd to run the assistant
# It loads environment variables and starts the main process

set -e

# Project paths
PROJECT_DIR="/Users/morgannstuart/Desktop/HomeAssistV2"
VENV_PYTHON="${PROJECT_DIR}/venv/bin/python"
ENV_FILE="${PROJECT_DIR}/.env"

# Change to project directory
cd "$PROJECT_DIR"

# Load environment variables from .env file
if [ -f "$ENV_FILE" ]; then
    set -a  # automatically export all variables
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# Verify venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: Python venv not found at $VENV_PYTHON"
    exit 1
fi

# Run the assistant
exec "$VENV_PYTHON" -m assistant_framework
